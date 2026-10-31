"""Small GPU ablation: V3 vs header randomization, same 8192-bit payload.

Use --plan-only for an offline plan. No performance improvement is assumed.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import random
import sys
import traceback
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import numpy as np

from core.randomized_header import RandomizedHeaderCore
from core.steganography import SteganographyCore
from run_comparison import aggregate, write_csv
from run_robustness_pilot import run_case, seed_everything, write_json

PROJECT = Path(__file__).resolve().parent
METHODS = ('v3', 'header_randomized')
PAYLOAD_BITS = 8192


class RecordingBackend:
    """Record latent arrays while delegating every model call to the same backend."""
    def __init__(self, backend):
        self.backend = backend
        self.directory = None
        self.inversions = 0

    def prepare(self, directory):
        self.directory = directory
        self.inversions = 0

    def generate(self, noise, context):
        if self.directory is not None:
            np.save(self.directory / 'initial_noise.npy', np.asarray(noise), allow_pickle=False)
        return self.backend.generate(noise, context)

    def invert(self, image, context):
        attacks = ('clean', 'jpeg', 'awgn', 'mblur')
        index = self.inversions
        self.inversions += 1
        noise = self.backend.invert(image, context)
        if self.directory is not None:
            name = attacks[index] if index < len(attacks) else str(index)
            np.save(self.directory / ('inverted_' + name + '.npy'), np.asarray(noise), allow_pickle=False)
        return noise


def load_plan(prompts_file, cases, seed):
    if type(cases) is not int or not 1 <= cases <= 5:
        raise ValueError('This development ablation supports 1..5 cases only')
    if type(seed) is not int or not 0 <= seed < 2 ** 32 - cases - 1000:
        raise ValueError('Invalid seed')
    prompts = [p.strip() for p in prompts_file.read_text(encoding='utf-8-sig').splitlines() if p.strip()]
    if len(prompts) < cases:
        raise ValueError('Not enough development prompts')
    return {'cases': cases, 'seed': seed, 'prompts': prompts[:cases], 'methods': list(METHODS),
            'payload_bits': PAYLOAD_BITS, 'header_bits': 1984, 'bpp': PAYLOAD_BITS / (512 * 512),
            'header_mask_seed': 20261028, 'generation_count': cases * 2,
            'inversion_count': cases * 2 * 4,
            'scope': 'Development ablation only. Shared 8192-bit messages and settings; different image-generating latent inputs.',
            'prompt_source_sha256': hashlib.sha256(prompts_file.read_bytes()).hexdigest(),
            'main_question': 'Does reversible header decorrelation improve actual image NIQE or extraction outcomes?',
            'caution': 'CPU distribution checks do not establish image performance. Include every failure; do not select only successful images.'}


def run_experiment(plan, cores, attack_fn, niqe_fn, scorer, output_root, *, seed_fn=seed_everything,
                   provenance=None):
    if set(cores) != set(METHODS):
        raise ValueError('Both V3 and randomized-header cores are required')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = output_root / ('run_' + stamp + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', dict(plan, provenance=provenance))
    write_json(output / 'status.json', {'status': 'running', 'completed_method_cases': 0})
    print('ABLATION DIRECTORY:', output.resolve(), flush=True)
    rows = []

    def save_summary(status):
        # Preserve scorer output as a log without confusing partial results with final scores.
        with redirect_stdout(io.StringIO()) as score_log:
            summaries = aggregate(rows, METHODS, scorer)
        (output / 'scorer_output.txt').write_text(score_log.getvalue(), encoding='utf-8')
        report = {'status': status, 'planned_cases_per_method': plan['cases'], 'methods': summaries,
                  'score_note': 'Course-PPT formula via B scorer, development set; not a final held-out result.'}
        if all(name in summaries for name in METHODS):
            a, b = summaries['v3'], summaries['header_randomized']
            report['candidate_minus_v3'] = {
                'mean_acc': {attack: b['mean_acc_including_failures'][attack] - a['mean_acc_including_failures'][attack]
                             for attack in ('clean', 'jpeg', 'awgn', 'mblur')},
                'mean_niqe': b['mean_metrics']['niqe'] - a['mean_metrics']['niqe']
                if a['niqe_complete'] and b['niqe_complete'] else None,
                'score': b['provisional_score'] - a['provisional_score']
                if a['provisional_score'] is not None and b['provisional_score'] is not None else None,
            }
        write_json(output / 'summary.json', report)
        return report

    try:
        for index, prompt in enumerate(plan['prompts']):
            seed = plan['seed'] + index
            rng = random.Random(seed)
            bits = ''.join(str(rng.randrange(2)) for _ in range(PAYLOAD_BITS))
            pair_contexts = {}
            for method in METHODS:
                directory = output / ('case_{:02d}_{}'.format(index, method))
                recorder = getattr(cores[method], '_backend', None)
                if isinstance(recorder, RecordingBackend):
                    recorder.prepare(directory)
                print('\nCASE {}/{} / {} / 8192 bits'.format(index + 1, plan['cases'], method), flush=True)
                row = run_case(cores[method], attack_fn, directory, prompt, bits, 3, seed, seed_fn)
                row.update(method=method, case_index=index, niqe=None, niqe_error=None,
                           payload_sha256=hashlib.sha256(bits.encode('ascii')).hexdigest())
                if row['generation_error'] is None:
                    pair_contexts[method] = json.loads((directory / 'context.json').read_text(encoding='utf-8'))
                    try:
                        with Image.open(directory / 'stego.png') as image:
                            value = float(niqe_fn(image.convert('RGB')))
                        if not math.isfinite(value) or value < 0:
                            raise ValueError('Invalid NIQE')
                        row['niqe'] = value
                        print('  NIQE: {:.6f}'.format(value), flush=True)
                    except Exception as exc:
                        row['niqe_error'] = str(exc)
                        (directory / 'niqe_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
                write_json(directory / 'result.json', row)
                rows.append(row)
                write_json(output / 'results.json', rows)
                write_csv(output / 'results.csv', rows)
                write_json(output / 'status.json', {'status': 'running', 'completed_method_cases': len(rows)})
            if len(pair_contexts) == 2:
                candidate = dict(pair_contexts['header_randomized'])
                if candidate.pop('header_transform', None) != 'public_xor_v1':
                    raise ValueError('Candidate did not use the declared header transform')
                if candidate.pop('header_mask_seed', None) != plan['header_mask_seed']:
                    raise ValueError('Candidate mask seed differs from the fixed plan')
                if candidate != pair_contexts['v3']:
                    raise ValueError('Generation/inversion contexts differ beyond the declared header transform')
            save_summary('running')
        report = save_summary('complete')
        incomplete = any(not item['niqe_complete'] or any(item['extraction_failures'].values())
                         for item in report['methods'].values())
        write_json(output / 'status.json', {'status': 'complete_with_failures' if incomplete else 'complete',
                                          'completed_method_cases': len(rows), 'exit_code': 1 if incomplete else 0})
    except Exception as exc:
        write_json(output / 'status.json', {'status': 'error', 'error': str(exc),
                                          'completed_method_cases': len(rows)})
        (output / 'experiment_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    print('\n=== HEADER ABLATION SUMMARY ===', flush=True)
    for method, item in report['methods'].items():
        print('{}: NIQE={}, score={}'.format(method, item['mean_metrics']['niqe'], item['provisional_score']))
        for attack in ('clean', 'jpeg', 'awgn', 'mblur'):
            print('  {}: Acc={:.4f}, failures={}/{}'.format(attack, item['mean_acc_including_failures'][attack],
                                                        item['extraction_failures'][attack], item['cases']))
    print('Candidate minus V3:', json.dumps(report.get('candidate_minus_v3'), ensure_ascii=False))
    print('Report:', (output / 'summary.json').resolve())
    return output, 1 if incomplete else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default=str(PROJECT / 'weights/v1-5-pruned.ckpt'))
    parser.add_argument('--clip', default=str(PROJECT / 'weights/clip'))
    parser.add_argument('--config', default=str(PROJECT / 'baseline/configs/stable-diffusion/ldm.yaml'))
    parser.add_argument('--prompts-file', type=Path, default=PROJECT / 'data/prompts/header_ablation_dev5.txt')
    parser.add_argument('--cases', type=int, default=5)
    parser.add_argument('--seed', type=int, default=73000)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'outputs/header-ablation')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.prompts_file, args.cases, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    for path in (Path(args.ckpt), Path(args.config)):
        if not path.is_file():
            parser.error('Missing local file: ' + str(path))
    if not Path(args.clip).is_dir():
        parser.error('Missing local CLIP directory')
    import torch
    if not torch.cuda.is_available():
        parser.error('This entry point requires a GPU. Use --plan-only for offline preparation.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['TORCH_HOME'] = str(PROJECT / 'weights/cache/torch')
    from evaluation.attacks import apply_attack
    from evaluation.scorer import calculate_final_score

    def niqe_fn(image):
        from evaluation.metrics import calculate_niqe
        return calculate_niqe(image)

    normal = SteganographyCore(args.ckpt, args.clip, args.config, device='cuda')
    candidate = RandomizedHeaderCore.from_backend(normal._backend, device=normal.device)
    shared_backend = normal._backend
    normal._backend = RecordingBackend(shared_backend)
    candidate._backend = RecordingBackend(shared_backend)
    names = ('core/mapping.py', 'core/steganography.py', 'core/randomized_header.py',
             'core/diffusion_backend.py', 'evaluation/attacks.py', 'evaluation/metrics.py',
             'evaluation/scorer.py', 'run_header_ablation.py', 'run_robustness_pilot.py', 'run_comparison.py')
    provenance = {
        'python': sys.version.split()[0],
        'saved_latents': 'Initial noise and completed inversion arrays are recorded for offline diagnosis; no reference message is used by the recorder.',
        'source_sha256': {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in names},
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy', 'Pillow', 'pyiqa')},
        'gpu': torch.cuda.get_device_name(0),
        'model_files': {'ckpt': str(Path(args.ckpt).resolve()), 'clip': str(Path(args.clip).resolve()),
                        'config': str(Path(args.config).resolve())},
        'attacks': {'jpeg_quality': 70, 'awgn_sigma_on_0_1': .05, 'median_kernel': 5},
    }
    _, code = run_experiment(plan, {'v3': normal, 'header_randomized': candidate},
                             apply_attack, niqe_fn, calculate_final_score, args.output_dir,
                             provenance=provenance)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
