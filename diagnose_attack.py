"""Diagnose saved v2 images; known-length payload accuracy is NOT an official score.

Does not generate images, alter attacks, correct payloads, or import evaluation/NIQE.
The reference is read only after image inversion and ordinary decoding.
"""
import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from core.mapping import SecretMapping
from core.steganography import SteganographyCore


def recover_symbols(mapping, noise):
    """Use the exact inverse transform of SecretMapping.decode, before thresholding."""
    noise = mapping._as_array(noise)
    unshuffled = noise.reshape(-1)[mapping._reverse_order].reshape(mapping.latent_shape)
    return (mapping._kernel.T @ unshuffled @ mapping._kernel) * 0.5 + 0.5


def compare_reference(mapping, symbols, reference):
    """Post-extraction diagnostic only: never passed back to the decoder."""
    if mapping.packet_version != 2:
        raise ValueError("This diagnostic supports packet version 2 only")
    if not reference or any(bit not in '01' for bit in reference):
        raise ValueError("Reference must be a nonempty binary string")
    # Encode only the expected packet symbols, not a new image or latent noise.
    expected = mapping.bits_to_tensor(reference).reshape(-1).astype(np.uint8)
    observed = (mapping._as_array(symbols).reshape(-1) > 0.5).astype(np.uint8)
    header = observed[:mapping.header_bits].reshape(mapping.HEADER_REPEATS, 64)
    expected_header = expected[:64]
    soft = (symbols.reshape(-1)[:mapping.header_bits].reshape(mapping.HEADER_REPEATS, 64).mean(axis=0) > 0.5).astype(np.uint8)
    wrong_per_column = (header != expected_header).sum(axis=0)
    majority = (header.sum(axis=0) > mapping.HEADER_REPEATS // 2).astype(np.uint8)
    payload = observed[mapping.header_bits:mapping.header_bits + len(reference)]
    expected_payload = expected[mapping.header_bits:mapping.header_bits + len(reference)]
    return {
        'reference_length': len(reference),
        'raw_header_errors': int(wrong_per_column.sum()),
        'raw_header_bits': mapping.header_bits,
        'header_errors_per_copy': [int(n) for n in (header != expected_header).sum(axis=1)],
        'max_errors_per_column': int(wrong_per_column.max()),
        'majority_length_errors': int((majority[:32] != expected_header[:32]).sum()),
        'majority_crc_errors': int((majority[32:] != expected_header[32:]).sum()),
        'soft_length_errors': int((soft[:32] != expected_header[:32]).sum()),
        'soft_crc_errors': int((soft[32:] != expected_header[32:]).sum()),
        'diagnostic_payload_errors': int((payload != expected_payload).sum()),
        'diagnostic_payload_acc_known_length': float((payload == expected_payload).mean()),
        'diagnostic_only': 'Known reference length bypasses the header for analysis, not scoring.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--config', default='baseline/configs/stable-diffusion/ldm.yaml')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--reference-bits-file', required=True, type=Path)
    metadata = parser.add_mutually_exclusive_group(required=True)
    metadata.add_argument('--context', type=Path)
    metadata.add_argument('--assume-v2-defaults', action='store_true',
                          help='Explicitly assume the last pipeline used v2 default parameters')
    parser.add_argument('--prompt', default='A photo of a cat')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.run_dir is None:
        candidates = [p for p in Path('outputs').glob('eval_header_v2_*') if p.is_dir()]
        if not candidates:
            parser.error('No outputs/eval_header_v2_* directory; provide --run-dir')
        args.run_dir = max(candidates, key=lambda p: p.name)
    names = ('original_stego.png', 'attacked_awgn.png')
    images = {}
    for name in names:
        with Image.open(args.run_dir / name) as image:
            if image.size != (512, 512):
                parser.error('Expected 512x512 image: ' + name)
            images[name] = image.convert('RGB').copy()
    if args.context:
        context = json.loads(args.context.read_text(encoding='utf-8-sig'))
    else:
        context = dict(SteganographyCore.DEFAULT_CONTEXT, prompt=args.prompt)
    context = SteganographyCore._validate_context(context)
    if context['version'] != 2:
        parser.error('Saved images must have been generated with version 2')
    print('Using saved run:', args.run_dir.resolve(), flush=True)
    print('Context source:', str(args.context) if args.context else 'ASSUMED v2 defaults; prompt=' + args.prompt,
          flush=True)
    core = SteganographyCore(args.ckpt, args.clip, args.config, device=args.device)
    mapping = core._mapping(context)
    observations = {}
    for name, image in images.items():
        symbols = recover_symbols(mapping, core._backend.invert(image, context))
        try:
            recovered = mapping.tensor_to_bits(symbols)
            error = None
        except ValueError as exc:
            recovered, error = None, str(exc)
        observations[name] = (symbols, recovered, error)
    # Original bits have not been read or used by either inversion or decoder.
    reference = args.reference_bits_file.read_text(encoding='utf-8-sig').strip()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.run_dir / ('diagnostic_' + stamp + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    clean = np.asarray(images[names[0]], dtype=np.float64) / 255.0
    attacked = np.asarray(images[names[1]], dtype=np.float64) / 255.0
    difference = attacked - clean
    report = {
        'context': context,
        'context_assumed': args.context is None,
        'run_dir': str(args.run_dir.resolve()),
        'awgn_observed_difference_mean_0_1': float(difference.mean()),
        'awgn_observed_difference_std_0_1': float(difference.std()),
        'noise_note': 'Saved pixel difference includes clipping and quantization; not the exact sampled sigma.',
        'images': {},
    }
    for name, (symbols, recovered, error) in observations.items():
        stats = compare_reference(mapping, symbols, reference)
        stats['decode_error'] = error
        stats['decoded_length'] = len(recovered) if recovered is not None else None
        stats['normal_extract_acc'] = (
            sum(a == b for a, b in zip(recovered, reference)) / len(reference)
            if recovered is not None and len(recovered) == len(reference) else 0.0
        )
        report['images'][name] = stats
        np.save(output / (name + '.symbols.npy'), symbols)
        if recovered is not None:
            (output / (name + '.recovered.txt')).write_text(recovered, encoding='utf-8')
        print('\n' + name)
        print('  Decoder:', error or 'OK')
        print('  Old hard-vote header errors: length={}, CRC={}'.format(
            stats['majority_length_errors'], stats['majority_crc_errors']))
        print('  Current mean header errors: length={}, CRC={}'.format(
            stats['soft_length_errors'], stats['soft_crc_errors']))
        print('  Payload diagnostic (KNOWN LENGTH, NOT SCORE): {}/{} errors, Acc={:.6f}'.format(
            stats['diagnostic_payload_errors'], len(reference), stats['diagnostic_payload_acc_known_length']))
        print('  Normal extraction Acc: {:.6f}'.format(stats['normal_extract_acc']))
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSaved-pixel noise std [0,1]: {:.6f}'.format(difference.std()))
    print('Diagnostic report:', (output / 'result.json').resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
