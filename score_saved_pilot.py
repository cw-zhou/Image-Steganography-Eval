"""Add NIQE and a provisional B-code score to saved pilot results; no generation."""
import argparse
import hashlib
import json
import math
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ATTACKS = ('jpeg', 'awgn', 'mblur')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def inspect_saved(run_dir, version):
    manifest = read_json(run_dir / 'manifest.json')
    if version not in manifest['versions']:
        raise ValueError('Requested version not present in the saved run')
    count = manifest['cases']
    if type(count) is not int or count < 1:
        raise ValueError('Invalid manifest case count')
    records = []
    for index in range(count):
        directory = run_dir / ('case_{:02d}_v{}'.format(index, version))
        result = read_json(directory / 'result.json')
        if result['packet_version'] != version:
            raise ValueError('Packet version mismatch: ' + str(directory))
        if result.get('generation_error') is not None:
            raise ValueError('Generation failed; cannot compute complete NIQE mean: ' + str(directory))
        reference = (directory / 'secret_bits.txt').read_text(encoding='utf-8-sig').strip()
        if not reference or any(c not in '01' for c in reference):
            raise ValueError('Invalid saved reference: ' + str(directory))
        if len(reference) != result['payload_bits'] or len(reference) != manifest['payload_bits']:
            raise ValueError('Payload length differs from recorded experiment')
        with Image.open(directory / 'stego.png') as image:
            if image.size != (512, 512):
                raise ValueError('Expected saved 512x512 stego image')
        accuracies = {}
        failures = {}
        for attack in ATTACKS:
            entry = result['attacks'][attack]  # Missing entries must not be silently skipped.
            decoded = entry['decoded'] and entry.get('error') is None
            if decoded:
                recovered = (directory / (attack + '_recovered.txt')).read_text(encoding='utf-8-sig').strip()
                if any(c not in '01' for c in recovered):
                    raise ValueError('Invalid recovered bit file')
                length_match = len(reference) == len(recovered)
                acc = sum(a == b for a, b in zip(reference, recovered)) / len(reference) if length_match else 0.0
            else:
                acc, length_match = 0.0, False
            reported = entry['acc']
            if not isinstance(reported, (int, float)) or not math.isfinite(reported) or abs(reported - acc) > 1e-10:
                raise ValueError('Stored accuracy disagrees with recovered files: {} {}'.format(directory, attack))
            accuracies[attack] = acc
            failures[attack] = not (decoded and length_match)
        records.append({'directory': directory, 'index': index, 'payload_bits': len(reference),
                        'bpp': len(reference) / (512 * 512), 'acc': accuracies, 'failures': failures})
    return manifest, records


def score_saved(run_dir, version, niqe_fn, score_fn, *, source_hashes=None):
    manifest, records = inspect_saved(run_dir, version)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = run_dir / ('score_' + stamp + '_' + uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    report = {
        'status': 'running', 'source_run': str(run_dir.resolve()),
        'packet_version': version, 'cases': len(records), 'payload_bits': manifest['payload_bits'],
        'attack_settings_from_manifest': manifest.get('attacks'), 'images': [],
        'score_scope': 'Provisional development score using B scorer; not verified official grading.',
        'aggregation': 'Apply B scorer once to mean bpp, each mean attack accuracy (failures included), and mean NIQE of original stego PNGs.',
        'evaluation_source_sha256': source_hashes,
    }
    path = output / 'result.json'
    write_json(path, report)
    try:
        for record in records:
            with Image.open(record['directory'] / 'stego.png') as image:
                niqe = float(niqe_fn(image.convert('RGB')))
            if not math.isfinite(niqe) or niqe < 0:
                raise ValueError('NIQE returned an invalid value')
            row = dict(record, directory=str(record['directory'].resolve()), niqe=niqe)
            report['images'].append(row)
            write_json(path, report)
            print('NIQE {}/{}: {:.6f}'.format(record['index'] + 1, len(records), niqe), flush=True)
        count = len(records)
        metrics = {
            'bpp': sum(r['bpp'] for r in records) / count,
            'niqe': sum(r['niqe'] for r in report['images']) / count,
            **{'acc_' + a: sum(r['acc'][a] for r in records) / count for a in ATTACKS},
        }
        print('\n=== PROVISIONAL SCORE: B formula, dataset mean metrics ===', flush=True)
        score = float(score_fn(**metrics))
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('Scorer returned a value outside 0..100')
        report.update(status='ok', mean_metrics=metrics, provisional_score=score,
                      extraction_failures={a: sum(r['failures'][a] for r in records) for a in ATTACKS})
        write_json(path, report)
    except Exception as exc:
        report.update(status='error', error=str(exc), provisional_score=None)
        write_json(path, report)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('Incomplete score; see:', path.resolve(), flush=True)
        raise
    print('Report:', path.resolve(), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--version', type=int, default=3, choices=(2, 3))
    args = parser.parse_args()
    # No diffusion backend, torch or pyiqa import until saved records have been checked.
    inspect_saved(args.run_dir, args.version)
    project = Path(__file__).resolve().parent
    os.environ['TORCH_HOME'] = str(project / 'weights/cache/torch')

    def niqe_fn(image):
        from evaluation.metrics import calculate_niqe
        return calculate_niqe(image)

    from evaluation.scorer import calculate_final_score
    source_hashes = {
        name: hashlib.sha256((project / name).read_bytes()).hexdigest()
        for name in ('evaluation/metrics.py', 'evaluation/scorer.py')
    }
    score_saved(args.run_dir, args.version, niqe_fn, calculate_final_score, source_hashes=source_hashes)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
