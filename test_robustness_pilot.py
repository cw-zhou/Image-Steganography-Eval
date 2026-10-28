import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from core.mapping import SecretMapping
from core.steganography import SteganographyCore
from run_robustness_pilot import run_case, summarize


class StrongHeaderTests(unittest.TestCase):
    def test_capacity_seed_and_independent_decode(self):
        mapping = SecretMapping(packet_version=3, padding_seed=42)
        self.assertEqual(mapping.header_bits, 1984)
        self.assertEqual(mapping.max_payload_bits, 14400)
        a = mapping.encode('01000')
        np.testing.assert_array_equal(a, mapping.encode('01000'))
        self.assertEqual(SecretMapping(packet_version=3).decode(a), '01000')
        other = SecretMapping(packet_version=3, padding_seed=43).encode('01000')
        self.assertFalse(np.array_equal(a, other))
        self.assertEqual(mapping.decode(mapping.encode('01' * 7200)), '01' * 7200)
        with self.assertRaises(ValueError):
            mapping.encode('1' * 14401)
        for seed in (-1, True, 2 ** 32):
            with self.assertRaises(ValueError):
                SecretMapping(padding_seed=seed)

    def test_fifteen_corrupt_copies_and_payload_error(self):
        mapping = SecretMapping(packet_version=3)
        symbols = mapping.bits_to_tensor('010000')
        flat = symbols.reshape(-1)
        copies = flat[:mapping.header_bits].reshape(31, 64)
        copies[:15] = 1 - copies[:15]
        flat[mapping.header_bits] = 1
        self.assertEqual(mapping.tensor_to_bits(symbols), '110000')
        copies[15, 31] = 1 - copies[15, 31]
        with self.assertRaisesRegex(ValueError, '校验失败'):
            mapping.tensor_to_bits(symbols)

    def test_v3_context_roundtrip_with_fresh_core(self):
        with patch('core.steganography.StableDiffusionBackend') as factory:
            backend = factory.return_value
            backend.generate.return_value = Image.new('RGB', (512, 512))
            core = SteganographyCore('unused', 'unused', 'unused', device='cpu')
            image = core.embed('10100', 'a cat', packet_version=3, padding_seed=42)
            noise, context = backend.generate.call_args.args
            self.assertEqual(context['version'], 3)
            self.assertNotIn('padding_seed', context)
            self.assertNotIn('secret_bits', context)
            backend.invert.return_value = noise
            fresh = SteganographyCore('unused', 'unused', 'unused', device='cpu')
            self.assertEqual(fresh.extract(image, context=json.loads(json.dumps(context))), '10100')
            self.assertEqual(core.DEFAULT_CONTEXT['version'], 2)


class PilotTests(unittest.TestCase):
    def make_core(self):
        core = Mock()
        core.embed.return_value = Image.new('RGB', (512, 512))
        core.get_context.return_value = dict(SteganographyCore.DEFAULT_CONTEXT, prompt='a cat', version=3)
        return core

    @patch('builtins.print')
    def test_saved_context_seeds_and_all_failure_modes_are_recorded(self, mock_print):
        core = self.make_core()
        core.extract.side_effect = ['0100', ValueError('CRC failure'), '0101', '0']
        seed_fn = Mock()
        attack = Mock(side_effect=lambda image, kind: image)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            directory = Path(tmp) / 'case'
            result = run_case(core, attack, directory, 'a cat', '0100', 3, 42, seed_fn)
            saved = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, result)
            self.assertEqual(result['attacks']['jpeg']['acc'], 0)
            self.assertFalse(result['attacks']['jpeg']['decoded'])
            self.assertEqual(result['attacks']['awgn']['acc'], 0.75)
            self.assertEqual(result['attacks']['mblur']['acc'], 0)
            self.assertTrue((directory / 'jpeg_traceback.txt').is_file())
            self.assertTrue((directory / 'context.json').is_file())
            core.embed.assert_called_once_with('0100', 'a cat', packet_version=3, padding_seed=42)
            for call in core.extract.call_args_list:
                self.assertEqual(set(call.kwargs), {'context'})
            self.assertEqual([call.args[0] for call in seed_fn.call_args_list],
                             [42, 142, 242, 143, 243, 144, 244, 145, 245])
            summary = summarize([result], [3])['3']['attacks']
            self.assertEqual(summary['jpeg']['failures'], 1)
            self.assertEqual(summary['awgn']['mean_acc_including_failures'], 0.75)
            self.assertEqual(summary['mblur']['failures'], 1)

    @patch('builtins.print')
    def test_generation_failure_counts_in_all_denominators(self, mock_print):
        core = self.make_core()
        core.embed.side_effect = RuntimeError('out of memory')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            result = run_case(core, Mock(), Path(tmp) / 'case', 'a cat', '01', 3, 42, Mock())
            core.extract.assert_not_called()
            summary = summarize([result], [2, 3])
            self.assertEqual(summary['2']['cases'], 0)
            self.assertEqual(summary['3']['generation_failures'], 1)
            for item in summary['3']['attacks'].values():
                self.assertEqual(item['failures'], 1)
                self.assertEqual(item['mean_acc_including_failures'], 0)


if __name__ == '__main__':
    unittest.main()
