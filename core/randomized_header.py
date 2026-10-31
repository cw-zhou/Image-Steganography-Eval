"""Experimental reversible header randomization; V3 and baseline remain unchanged.

Public XOR masking removes the repeated raw header pattern before orthogonal
mapping. It is neither encryption nor additional error correction. Image quality
and attack performance have not been validated on a GPU.
"""
from typing import Optional

import numpy as np

from .mapping import SecretMapping
from .steganography import SteganographyCore


class RandomizedHeaderMapping(SecretMapping):
    def __init__(self, *, header_mask_seed: int = 20261028, **kwargs):
        version = kwargs.pop('packet_version', 3)
        if type(version) is not int or version != 3:
            raise ValueError('Randomized header candidate requires V3 framing')
        if type(header_mask_seed) is not int or not 0 <= header_mask_seed < 2 ** 32:
            raise ValueError('header_mask_seed must be a uint32 integer')
        super().__init__(packet_version=3, **kwargs)
        self.header_mask_seed = header_mask_seed
        self._header_mask = np.random.default_rng(header_mask_seed).integers(
            0, 2, self.header_bits, dtype=np.uint8)

    def bits_to_tensor(self, secret_bits: str) -> np.ndarray:
        symbols = super().bits_to_tensor(secret_bits)
        flat = symbols.reshape(-1)
        flat[:self.header_bits] = np.where(
            self._header_mask, 1.0 - flat[:self.header_bits], flat[:self.header_bits])
        return symbols

    def tensor_to_bits(self, tensor, threshold: float = 0.5) -> str:
        # Reflect continuous observations BEFORE combining repetitions. Do not
        # threshold or clip first: negative or >1 values may carry confidence.
        symbols = self._as_array(tensor).copy()
        flat = symbols.reshape(-1)
        flat[:self.header_bits] = np.where(
            self._header_mask, 1.0 - flat[:self.header_bits], flat[:self.header_bits])
        return super().tensor_to_bits(symbols, threshold=threshold)


class RandomizedHeaderCore(SteganographyCore):
    """Separate opt-in core; its saved context cannot be mistaken for ordinary V3."""
    DEFAULT_CONTEXT = dict(SteganographyCore.DEFAULT_CONTEXT, version=3,
                           header_transform='public_xor_v1', header_mask_seed=20261028)

    @classmethod
    def from_backend(cls, backend, *, device="cuda"):
        """Share a loaded model for an ablation without loading a second SD copy."""
        if not callable(getattr(backend, "generate", None)) or not callable(getattr(backend, "invert", None)):
            raise TypeError("Backend must implement generate and invert")
        instance = cls.__new__(cls)
        instance.device = str(device)
        instance._context = None
        instance._backend = backend
        return instance

    @classmethod
    def _validate_context(cls, context: dict) -> dict:
        result = super()._validate_context(context)
        if result['version'] != 3 or result['header_transform'] != 'public_xor_v1':
            raise ValueError('RandomizedHeaderCore context identity mismatch')
        seed = result['header_mask_seed']
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError('header_mask_seed must be a uint32 integer')
        return result

    @staticmethod
    def _mapping(context: dict, *, padding_seed: Optional[int] = None) -> RandomizedHeaderMapping:
        return RandomizedHeaderMapping(
            seed_kernel=context['seed_kernel'], seed_shuffle=context['seed_shuffle'],
            padding_seed=padding_seed, header_mask_seed=context['header_mask_seed'])
