"""Native fixed-length baseline mapping with the shared diffusion backend.

The original ours_mapping(bits=1) carries all 16384 symbols without a header.
This controlled adapter is not a byte-for-byte reproduction of txt2img.py.
"""
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from .steganography import SteganographyCore


def load_original_mapping():
    path = Path(__file__).resolve().parent.parent / 'baseline/scripts/mapping_module.py'
    spec = importlib.util.spec_from_file_location('_stego_baseline_reference_mapping', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ours_mapping(bits=1)


class BaselineReferenceCore:
    CAPACITY = 4 * 64 * 64
    FRAMING = 'baseline_fixed_16384_no_header'

    def __init__(self, backend):
        self._backend = backend
        self._mapping = load_original_mapping()
        self._context = None

    @staticmethod
    def _call_without_rng_side_effect(fn, *args, **kwargs):
        state = np.random.get_state()
        try:
            return fn(*args, **kwargs)
        finally:
            np.random.set_state(state)

    def embed(self, secret_bits, prompt, *, packet_version=None, padding_seed=None):
        if packet_version not in (None, 'baseline'):
            raise ValueError('Native baseline uses fixed framing, not a versioned length header')
        if not isinstance(secret_bits, str) or len(secret_bits) != self.CAPACITY or any(c not in '01' for c in secret_bits):
            raise ValueError('Native single-bit baseline requires exactly 16384 binary bits')
        context = dict(SteganographyCore.DEFAULT_CONTEXT, prompt=prompt)
        context = SteganographyCore._validate_context(context)
        context['framing'] = self.FRAMING
        symbols = (np.frombuffer(secret_bits.encode('ascii'), dtype=np.uint8) - 48).reshape(1, 4, 64, 64)
        noise = self._call_without_rng_side_effect(
            self._mapping.encode_secret, symbols,
            seed_kernel=context['seed_kernel'], seed_shuffle=context['seed_shuffle'])
        image = self._backend.generate(np.asarray(noise, dtype=np.float32), context)
        if not isinstance(image, Image.Image) or image.size != (512, 512) or image.mode != 'RGB':
            raise RuntimeError('Backend must return a 512x512 RGB image')
        self._context = deepcopy(context)
        return image

    def get_context(self):
        if self._context is None:
            raise RuntimeError('No context; embed first or provide saved context')
        return deepcopy(self._context)

    def extract(self, image, *, context=None):
        context = deepcopy(context) if context is not None else self.get_context()
        if not isinstance(context, dict) or context.pop('framing', None) != self.FRAMING:
            raise ValueError('Native baseline framing context required')
        context = SteganographyCore._validate_context(context)
        if not isinstance(image, Image.Image) or image.size != (512, 512):
            raise ValueError('Expected 512x512 image')
        noise = self._backend.invert(image.convert('RGB'), context)
        noise = np.asarray(noise)
        if noise.shape != (1, 4, 64, 64) or not np.isfinite(noise).all():
            raise ValueError('Invalid inversion output')
        recovered = self._call_without_rng_side_effect(
            self._mapping.decode_secret, noise,
            seed_kernel=context['seed_kernel'], seed_shuffle=context['seed_shuffle'])
        if recovered.shape != noise.shape or not np.isfinite(recovered).all() or not np.isin(recovered, [0, 1]).all():
            raise ValueError('Original baseline mapping did not produce binary symbols')
        return ''.join(str(int(bit)) for bit in recovered.reshape(-1))
