"""Fixed-seed, small-sample comparison of v2/v3 header protection. Not final grading."""
import argparse
import importlib.metadata
import json
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from core.mapping import SecretMapping
from core.steganography import SteganographyCore

PROMPTS = (
    'A photo of a cat sitting on a wooden chair',
    'A mountain landscape with a lake under a blue sky',
    'A photo of a city street with buildings in daylight',
)
ATTACKS = ('clean', 'jpeg', 'awgn', 'mblur')


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def accuracy(reference, recovered):
    if not reference or len(reference) != len(recovered):
        return 0.0
    return sum(a == b for a, b in zip(reference, recovered)) / len(reference)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_case(core, attack_fn, directory, prompt, bits, version, seed, seed_fn=seed_everything, *, header_bits=None):
    directory.mkdir(parents=True, exist_ok=False)
    header = SecretMapping(packet_version=version).header_bits if header_bits is None else header_bits
    result = {
        'prompt': prompt, 'seed': seed, 'packet_version': version,
        'payload_bits': len(bits), 'header_bits': header,
        'bpp': len(bits) / (512 * 512), 'attacks': {},
        'generation_error': None,
    }
    (directory / 'secret_bits.txt').write_text(bits, encoding='utf-8')
    try:
        seed_fn(seed)
        started = time.perf_counter()
        image = core.embed(bits, prompt, packet_version=version, padding_seed=seed)
        context = core.get_context()
        write_json(directory / 'context.json', context)
        image.save(directory / 'stego.png')
        with Image.open(directory / 'stego.png') as stored:
            image = stored.convert('RGB').copy()
        result['generation_seconds'] = time.perf_counter() - started
    except Exception as exc:
        result['generation_error'] = str(exc)
        (directory / 'generation_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(directory / 'result.json', result)
        return result
    for index, attack in enumerate(ATTACKS):
        attack_seed = seed + 100 + index
        inversion_seed = seed + 200 + index
        entry = {
            'acc': 0.0, 'decoded': False, 'length_match': False,
            'exact_match': False, 'error': None,
            'attack_seed': attack_seed, 'inversion_seed': inversion_seed,
        }
        started = time.perf_counter()
        try:
            seed_fn(attack_seed)
            attacked = image.copy() if attack == 'clean' else attack_fn(image.copy(), attack)
            attacked.save(directory / (attack + '.png'))
            with Image.open(directory / (attack + '.png')) as stored:
                attacked = stored.convert('RGB').copy()
            seed_fn(inversion_seed)
            recovered = core.extract(attacked, context=context)
            if not isinstance(recovered, str) or any(c not in '01' for c in recovered):
                raise ValueError('Decoder did not return a binary string')
            (directory / (attack + '_recovered.txt')).write_text(recovered, encoding='utf-8')
            entry.update(decoded=True, decoded_length=len(recovered),
                         length_match=len(bits) == len(recovered),
                         acc=accuracy(bits, recovered), exact_match=bits == recovered)
        except Exception as exc:
            entry['error'] = str(exc)
            (directory / (attack + '_traceback.txt')).write_text(traceback.format_exc(), encoding='utf-8')
        entry['seconds'] = time.perf_counter() - started
        result['attacks'][attack] = entry
        write_json(directory / 'result.json', result)
        print('  {}: Acc={:.4f}, decoded={}, error={}'.format(
            attack, entry['acc'], entry['decoded'], entry['error']), flush=True)
    return result


def summarize(results, versions):
    summary = {}
    for version in versions:
        rows = [r for r in results if r['packet_version'] == version]
        count = len(rows)
        item = {'cases': count, 'generation_failures': sum(r['generation_error'] is not None for r in rows),
                'attacks': {}}
        for attack in ATTACKS:
            entries = [r['attacks'].get(attack, {}) for r in rows]
            item['attacks'][attack] = {
                'mean_acc_including_failures': sum(e.get('acc', 0.0) for e in entries) / count if count else None,
                'successful_decodes': sum(bool(e.get('decoded') and e.get('length_match')) for e in entries),
                'failures': sum(not (e.get('decoded') and e.get('length_match')) for e in entries),
                'exact_matches': sum(bool(e.get('exact_match')) for e in entries),
            }
        summary[str(version)] = item
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default='weights/v1-5-pruned.ckpt')
    parser.add_argument('--clip', default='weights/clip')
    parser.add_argument('--config', default='baseline/configs/stable-diffusion/ldm.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cases', type=int, default=3)
    parser.add_argument('--bit-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--versions', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--prompts-file', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/robustness-pilot'))
    args = parser.parse_args(argv)
    if args.cases < 1 or not 0 <= args.seed <= 2 ** 32 - 1000 - args.cases:
        parser.error('Invalid cases or seed')
    if not args.versions or len(set(args.versions)) != len(args.versions) or any(v not in (2, 3) for v in args.versions):
        parser.error('versions must be distinct entries from 2, 3')
    capacity = min(SecretMapping(packet_version=v).max_payload_bits for v in args.versions)
    if not 1 <= args.bit_length <= capacity:
        parser.error('bit-length must be within 1..{}'.format(capacity))
    prompts = list(PROMPTS)
    if args.prompts_file:
        prompts = [line.strip() for line in args.prompts_file.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(prompts) < args.cases:
        parser.error('Need at least --cases prompts; supply --prompts-file for more than 3')
    # Import only the attack module. Avoid importing metrics, NIQE, or a scorer.
    from evaluation.attacks import apply_attack
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir / ('run_' + stamp + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print('PILOT DIRECTORY:', output.resolve(), flush=True)
    manifest = {
        'versions': args.versions, 'cases': args.cases, 'seed': args.seed,
        'payload_bits': args.bit_length, 'prompts': prompts[:args.cases],
        'attacks': {'jpeg_quality': 70, 'awgn_sigma_on_0_1': 0.05, 'median_kernel': 5},
        'python': sys.version.split()[0],
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'scipy', 'Pillow')},
        'scope': 'Development pilot, not 100-prompt final scoring; no NIQE or final score.',
        'reproducibility': 'All message/padding/attack/inversion seeds are recorded. Bitwise equality across hardware or library versions is not guaranteed.',
        'comparison': 'Same prompts, payloads and random seeds; packet formats differ, so generated images differ.',
        'warning': 'CRC protects length only. A successful decode does not mean all payload bits are correct.',
    }
    write_json(output / 'manifest.json', manifest)
    try:
        core = SteganographyCore(args.ckpt, args.clip, args.config, device=args.device)
    except Exception:
        (output / 'model_load_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    results = []
    for i, prompt in enumerate(prompts[:args.cases]):
        seed = args.seed + i
        rng = random.Random(seed)
        bits = ''.join(str(rng.randrange(2)) for _ in range(args.bit_length))
        for version in args.versions:
            print('\nCASE {}, VERSION {}, SEED {}'.format(i + 1, version, seed), flush=True)
            result = run_case(core, apply_attack, output / ('case_{:02d}_v{}'.format(i, version)),
                              prompt, bits, version, seed)
            results.append(result)
            if result['generation_error']:
                print('  Generation failed:', result['generation_error'], flush=True)
            write_json(output / 'results.json', results)
            write_json(output / 'summary.json', summarize(results, args.versions))
    summary = summarize(results, args.versions)
    print('\n=== PILOT SUMMARY (failures count as zero accuracy) ===')
    for version in args.versions:
        print('Version {}:'.format(version))
        for attack, stats in summary[str(version)]['attacks'].items():
            print('  {}: decoded {}/{}, mean Acc={:.4f}'.format(
                attack, stats['successful_decodes'], args.cases, stats['mean_acc_including_failures']))
    print('Summary:', (output / 'summary.json').resolve())
    # Failures are data, not hidden: nonzero exit if any case/attack failed.
    failed = any(s['failures'] for v in summary.values() for s in v['attacks'].values())
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
