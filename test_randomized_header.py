import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from core.mapping import SecretMapping
from core.randomized_header import RandomizedHeaderMapping, RandomizedHeaderCore
from core.steganography import SteganographyCore


class RandomizedMappingTests(unittest.TestCase):
    def test_independent_decoder_for_short_and_full_length_messages(self):
        encoder = RandomizedHeaderMapping(padding_seed=42)
        for bits in ('0', '1', '101000', '01' * 4096, '10' * 7200):
            with self.subTest(length=len(bits)):
                self.assertEqual(RandomizedHeaderMapping().decode(encoder.encode(bits)), bits)
        self.assertEqual(encoder.header_bits, 1984)
        self.assertEqual(encoder.max_payload_bits, 14400)
        with self.assertRaises(ValueError):
            encoder.encode('0' * 14401)

    def test_only_header_changes_with_same_padding_seed(self):
        normal = SecretMapping(packet_version=3, padding_seed=42).bits_to_tensor('101000')
        masked = RandomizedHeaderMapping(padding_seed=42).bits_to_tensor('101000')
        np.testing.assert_array_equal(normal.reshape(-1)[1984:], masked.reshape(-1)[1984:])
        self.assertFalse(np.array_equal(normal, masked))
        self.assertEqual(len(np.unique(normal.reshape(-1)[:1984].reshape(31, 64), axis=0)), 1)
        self.assertEqual(len(np.unique(masked.reshape(-1)[:1984].reshape(31, 64), axis=0)), 31)

    def test_header_protection_survives_masking_and_still_rejects_damage(self):
        mapping = RandomizedHeaderMapping(padding_seed=42)
        symbols = mapping.bits_to_tensor('010000')
        copies = symbols.reshape(-1)[:1984].reshape(31, 64)
        copies[:15] = 1 - copies[:15]
        self.assertEqual(mapping.tensor_to_bits(symbols), '010000')
        copies[15, 31] = 1 - copies[15, 31]
        with self.assertRaisesRegex(ValueError, '校验失败'):
            mapping.tensor_to_bits(symbols)

    def test_continuous_confidence_and_input_immutability(self):
        mapping = RandomizedHeaderMapping(padding_seed=42)
        symbols = mapping.bits_to_tensor('010000')
        # In the raw masked domain, extend 0/1 to -0.1/1.1: no clipping needed.
        symbols = symbols * 1.2 - .1
        before = symbols.copy()
        self.assertEqual(mapping.tensor_to_bits(symbols), '010000')
        np.testing.assert_array_equal(before, symbols)

    def test_payload_error_is_not_replaced_by_correct_reference(self):
        mapping = RandomizedHeaderMapping(padding_seed=42)
        symbols = mapping.bits_to_tensor('010000')
        symbols.reshape(-1)[1984] = 1
        self.assertEqual(RandomizedHeaderMapping().tensor_to_bits(symbols), '110000')

    def test_mask_seed_reproducibility_validation_and_rng_isolation(self):
        before = np.random.get_state()
        first = RandomizedHeaderMapping(padding_seed=42).encode('1000')
        second = RandomizedHeaderMapping(padding_seed=42).encode('1000')
        np.testing.assert_array_equal(first, second)
        with self.assertRaises(ValueError):
            RandomizedHeaderMapping(header_mask_seed=1).decode(first)
        for value in (-1, True, 2 ** 32):
            with self.assertRaises(ValueError):
                RandomizedHeaderMapping(header_mask_seed=value)
        with self.assertRaises(ValueError):
            RandomizedHeaderMapping(packet_version=2)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


class CandidateCoreTests(unittest.TestCase):
    def test_context_identity_fresh_instance_and_no_secret_in_context(self):
        with patch('core.steganography.StableDiffusionBackend') as factory:
            backend = factory.return_value
            backend.generate.return_value = Image.new('RGB', (512, 512))
            core = RandomizedHeaderCore('unused', 'unused', 'unused', device='cpu')
            image = core.embed('10000', 'a cat', padding_seed=42)
            noise, context = backend.generate.call_args.args
            context = json.loads(json.dumps(context))
            self.assertEqual(context['header_transform'], 'public_xor_v1')
            self.assertNotIn('secret_bits', context)
            self.assertNotIn('message_length', context)
            self.assertNotIn('padding_seed', context)
            backend.invert.return_value = noise
            fresh = RandomizedHeaderCore('unused', 'unused', 'unused', device='cpu')
            self.assertEqual(fresh.extract(image, context=context), '10000')
            with self.assertRaises(ValueError):
                SteganographyCore._validate_context(context)
            with self.assertRaises(ValueError):
                RandomizedHeaderCore._validate_context(dict(SteganographyCore.DEFAULT_CONTEXT, prompt='a cat', version=3))
            invalid = dict(context, header_transform='other')
            with self.assertRaises(ValueError):
                fresh.extract(image, context=invalid)
            invalid = dict(context, header_mask_seed=True)
            with self.assertRaises(ValueError):
                fresh.extract(image, context=invalid)
            self.assertEqual(SteganographyCore.DEFAULT_CONTEXT['version'], 2)


if __name__ == '__main__':
    unittest.main()
