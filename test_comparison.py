import ast
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from scipy.linalg import qr

from core.baseline_reference import BaselineReferenceCore
from run_comparison import aggregate, write_csv
from run_robustness_pilot import run_case


def original_mapping_for_cpu_test():
    # Execute the author's unmodified mapping classes, without its unused torch import.
    source = Path(__file__).parent / 'baseline/scripts/mapping_module.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('mapping_module', 'ours_mapping')]
    namespace = {'np': np, 'qr': qr}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['ours_mapping'](bits=1)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.loader = patch('core.baseline_reference.load_original_mapping', side_effect=original_mapping_for_cpu_test)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.backend = Mock()
        self.backend.generate.return_value = Image.new('RGB', (512, 512))
        self.core = BaselineReferenceCore(self.backend)

    def test_original_fixed_mapping_and_fresh_decode_without_message(self):
        message = '0010' * 4096
        before = np.random.get_state()
        image = self.core.embed(message, 'a cat', packet_version='baseline', padding_seed=42)
        noise, context = self.backend.generate.call_args.args
        self.assertEqual(noise.shape, (1, 4, 64, 64))
        self.assertEqual(noise.dtype, np.float32)
        self.assertNotIn('secret_bits', context)
        self.assertNotIn('message_length', context)
        self.backend.invert.return_value = noise
        fresh = BaselineReferenceCore(self.backend)
        self.assertEqual(fresh.extract(image, context=self.core.get_context()), message)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_no_reference_fallback_and_validation(self):
        image = self.core.embed('1' * 16384, 'a cat')
        original = original_mapping_for_cpu_test()
        self.backend.invert.return_value = original.encode_secret(
            np.zeros((1, 4, 64, 64)), seed_kernel=100, seed_shuffle=101).astype(np.float32)
        self.assertEqual(self.core.extract(image), '0' * 16384)
        with self.assertRaises(ValueError):
            self.core.embed('1' * 8192, 'a cat')
        context = self.core.get_context()
        context['framing'] = 'v3'
        with self.assertRaises(ValueError):
            self.core.extract(image, context=context)
        self.backend.invert.return_value = np.full((1, 4, 64, 64), np.nan)
        with self.assertRaises(ValueError):
            self.core.extract(image)

    @patch('builtins.print')
    def test_native_runner_uses_zero_header_and_full_payload(self, mock_print):
        self.backend.invert.side_effect = lambda image, ctx: self.backend.generate.call_args.args[0]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            result = run_case(self.core, lambda im, kind: im, Path(tmp) / 'case', 'a cat',
                              '01' * 8192, 'baseline', 42, Mock(), header_bits=0)
        self.assertEqual(result['header_bits'], 0)
        self.assertEqual(result['bpp'], .0625)
        self.assertTrue(all(row['exact_match'] for row in result['attacks'].values()))


class ComparisonAggregateTests(unittest.TestCase):
    def row(self, method, failure=False, niqe=3.0):
        return {'method': method, 'bpp': .0625 if method == 'baseline' else .03125,
                'niqe': niqe, 'generation_error': None,
                'attacks': {a: {'acc': 0 if failure else .9, 'decoded': not failure,
                                'length_match': not failure} for a in ('clean', 'jpeg', 'awgn', 'mblur')}}

    def test_native_and_v3_capacities_not_conflated(self):
        scorer = Mock(return_value=70)
        result = aggregate([self.row('baseline'), self.row('v3')], ['baseline', 'v3'], scorer)
        self.assertEqual(result['baseline']['mean_metrics']['bpp'], .0625)
        self.assertEqual(result['v3']['mean_metrics']['bpp'], .03125)
        self.assertEqual(scorer.call_count, 2)

    def test_failures_count_and_missing_niqe_blocks_score(self):
        scorer = Mock(return_value=50)
        result = aggregate([self.row('v3'), self.row('v3', failure=True)], ['v3'], scorer)['v3']
        self.assertEqual(result['mean_acc_including_failures']['awgn'], .45)
        self.assertEqual(result['extraction_failures']['awgn'], 1)
        scorer.reset_mock()
        result = aggregate([self.row('v3'), self.row('v3', niqe=None)], ['v3'], scorer)['v3']
        self.assertIsNone(result['provisional_score'])
        scorer.assert_not_called()

    def test_csv_preserves_failure_and_metric(self):
        row = self.row('v3', failure=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / 'results.csv'
            write_csv(path, [row])
            with path.open(encoding='utf-8-sig', newline='') as file:
                saved = next(csv.DictReader(file))
            self.assertEqual(saved['decoded_awgn'], 'False')
            self.assertEqual(saved['acc_awgn'], '0')


if __name__ == '__main__':
    unittest.main()
