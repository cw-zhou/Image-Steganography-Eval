"""Controlled baseline-native vs v3 benchmark. Scores are provisional B-formula scores."""
import argparse
import csv
import hashlib
import importlib.metadata
import math
import os
import random
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from core.baseline_reference import BaselineReferenceCore
from core.mapping import SecretMapping
from core.steganography import SteganographyCore
from run_robustness_pilot import run_case, seed_everything, write_json

ATTACKS = ('clean', 'jpeg', 'awgn', 'mblur')


def aggregate(rows, methods, scorer):
    summary = {}
    for method in methods:
        group = [row for row in rows if row['method'] == method]
        count = len(group)
        if not count:
            continue
        means = {a: sum(r['attacks'].get(a, {}).get('acc', 0.0) for r in group) / count for a in ATTACKS}
        failures = {a: sum(not (r['attacks'].get(a, {}).get('decoded') and r['attacks'].get(a, {}).get('length_match')) for r in group) for a in ATTACKS}
        niqes = [r.get('niqe') for r in group]
        complete = all(isinstance(n, (int, float)) and math.isfinite(n) and n >= 0 for n in niqes)
        complete = complete and all(r['generation_error'] is None for r in group)
        metrics = dict(bpp=sum(r['bpp'] for r in group) / count,
                       acc_jpeg=means['jpeg'], acc_awgn=means['awgn'], acc_mblur=means['mblur'],
                       niqe=sum(niqes) / count if complete else None)
        score = float(scorer(**metrics)) if complete else None
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 100):
            raise ValueError('Scorer returned an invalid value')
        summary[method] = {'cases': count, 'mean_acc_including_failures': means,
                           'extraction_failures': failures, 'mean_metrics': metrics,
                           'provisional_score': score, 'niqe_complete': complete}
    return summary


def write_csv(path, rows):
    fields = ['method', 'seed', 'prompt', 'payload_bits', 'header_bits', 'bpp', 'niqe', 'generation_error', 'niqe_error']
    fields += ['acc_' + a for a in ATTACKS]
    fields += ['decoded_' + a for a in ATTACKS]
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = {k: row.get(k) for k in fields[:10]}
            for attack in ATTACKS:
                entry = row['attacks'].get(attack, {})
                data['acc_' + attack] = entry.get('acc', 0.0)
                data['decoded_' + attack] = bool(entry.get('decoded') and entry.get('length_match'))
            writer.writerow(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default='weights/v1-5-pruned.ckpt')
    parser.add_argument('--clip', default='weights/clip')
    parser.add_argument('--config', default='baseline/configs/stable-diffusion/ldm.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prompts-file', type=Path, default=Path('weights/pilot_prompts_10.txt'))
    parser.add_argument('--cases', type=int, default=3)
    parser.add_argument('--seed', type=int, default=314)
    parser.add_argument('--bit-length', type=int, default=8192, help='V3 payload; native baseline always carries 16384 bits')
    parser.add_argument('--methods', nargs='+', default=['baseline', 'v3'], choices=['baseline', 'v3'])
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/comparison'))
    args = parser.parse_args(argv)
    if args.cases < 1 or not 0 <= args.seed <= 2 ** 32 - 1000 - args.cases:
        parser.error('Invalid cases or seed')
    if len(set(args.methods)) != len(args.methods):
        parser.error('Duplicate methods')
    if not 1 <= args.bit_length <= SecretMapping(packet_version=3).max_payload_bits:
        parser.error('V3 bit-length must be in 1..14400')
    prompts = [line.strip() for line in args.prompts_file.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(prompts) < args.cases:
        parser.error('Not enough prompts')
    project = Path(__file__).resolve().parent
    os.environ['TORCH_HOME'] = str(project / 'weights/cache/torch')
    from evaluation.attacks import apply_attack
    from evaluation.scorer import calculate_final_score
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir / ('run_' + stamp + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print('COMPARISON DIRECTORY:', output.resolve(), flush=True)
    manifest = {
        'cases': args.cases, 'seed': args.seed, 'methods': args.methods, 'prompts': prompts[:args.cases],
        'baseline_payload_bits': 16384, 'v3_payload_bits': args.bit_length,
        'baseline_framing': 'Original ours_mapping(bits=1), fixed 16384-bit payload, no header or filler.',
        'v3_framing': 'Variable length, 31 repeated length+CRC headers; payload decoded without the reference.',
        'comparison': 'Same diffusion backend, prompts, steps, seeds, attack code, NIQE, scorer. Different capacities and framing; NOT an equal-payload ablation.',
        'backend_note': 'Shared backend normalizes and saves PNG as implemented in core/diffusion_backend.py; not a byte-for-byte replay of original txt2img.py.',
        'scope': 'Controlled development comparison. Final 100-prompt test must use a held-out set. B grading formula remains provisional.',
        'attacks': {'jpeg_quality': 70, 'awgn_sigma_on_0_1': .05, 'median_kernel': 5},
        'python': sys.version.split()[0],
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'scipy', 'Pillow', 'pyiqa')},
        'model_files': {'ckpt': str(Path(args.ckpt).resolve()), 'clip': str(Path(args.clip).resolve()), 'config': str(Path(args.config).resolve())},
        'source_sha256': {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in (
            'baseline/scripts/mapping_module.py', 'core/baseline_reference.py', 'core/mapping.py',
            'core/steganography.py', 'core/diffusion_backend.py', 'evaluation/attacks.py',
            'evaluation/metrics.py', 'evaluation/scorer.py', 'run_robustness_pilot.py', 'run_comparison.py')},
    }
    write_json(output / 'manifest.json', manifest)
    try:
        improved = SteganographyCore(args.ckpt, args.clip, args.config, device=args.device)
        cores = {'v3': improved}
        if 'baseline' in args.methods:
            cores['baseline'] = BaselineReferenceCore(improved._backend)
    except Exception:
        (output / 'model_load_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    rows = []
    for index, prompt in enumerate(prompts[:args.cases]):
        seed = args.seed + index
        rng = random.Random(seed)
        full_bits = ''.join(str(rng.randrange(2)) for _ in range(16384))
        for method in args.methods:
            bits = full_bits if method == 'baseline' else full_bits[:args.bit_length]
            version = 'baseline' if method == 'baseline' else 3
            directory = output / ('case_{:02d}_{}'.format(index, method))
            print('\nCASE {} / {} / {} payload bits'.format(index + 1, method, len(bits)), flush=True)
            row = run_case(cores[method], apply_attack, directory, prompt, bits, version, seed,
                           header_bits=0 if method == 'baseline' else None)
            row.update(method=method, niqe=None, niqe_error=None)
            if row['generation_error'] is None:
                try:
                    from evaluation.metrics import calculate_niqe
                    with Image.open(directory / 'stego.png') as image:
                        niqe = float(calculate_niqe(image.convert('RGB')))
                    if not math.isfinite(niqe) or niqe < 0:
                        raise ValueError('Invalid NIQE')
                    row['niqe'] = niqe
                    print('  NIQE: {:.6f}'.format(niqe), flush=True)
                except Exception as exc:
                    row['niqe_error'] = str(exc)
                    (directory / 'niqe_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
            write_json(directory / 'result.json', row)
            rows.append(row)
            write_json(output / 'results.json', rows)
            write_csv(output / 'results.csv', rows)
    print('\n=== COMPARISON SUMMARY: provisional B-formula scores ===', flush=True)
    summary = aggregate(rows, args.methods, calculate_final_score)
    write_json(output / 'summary.json', summary)
    for method, data in summary.items():
        print('\n{}: bpp={:.6f}, NIQE={}, provisional score={}'.format(
            method, data['mean_metrics']['bpp'], data['mean_metrics']['niqe'], data['provisional_score']))
        for attack in ATTACKS:
            print('  {}: Acc={:.4f}, failures={}/{}'.format(attack, data['mean_acc_including_failures'][attack],
                                                        data['extraction_failures'][attack], data['cases']))
    print('Summary:', (output / 'summary.json').resolve())
    incomplete = any(not data['niqe_complete'] or any(data['extraction_failures'].values()) for data in summary.values())
    return 1 if incomplete else 0


if __name__ == '__main__':
    raise SystemExit(main())
