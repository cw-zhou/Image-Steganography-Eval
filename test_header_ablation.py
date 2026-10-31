import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import numpy as np

from core.randomized_header import RandomizedHeaderCore
from core.steganography import SteganographyCore
from run_header_ablation import RecordingBackend, load_plan, main, run_experiment


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prompts = self.root / 'dev.txt'
        self.prompts.write_text('a cat\na dog\na mountain\na beach\na forest\n', encoding='utf-8')
        self.plan = load_plan(self.prompts, 1, 73000)

    def cores(self):
        contexts = {
            'v3': dict(SteganographyCore.DEFAULT_CONTEXT, version=3, prompt='a cat'),
            'header_randomized': dict(RandomizedHeaderCore.DEFAULT_CONTEXT, prompt='a cat'),
        }
        result = {}
        for name, context in contexts.items():
            core = Mock()
            core.embed.return_value = Image.new('RGB', (512, 512))
            core.get_context.return_value = context
            core.extract.return_value = '0' * 8192
            result[name] = core
        return result

    def invoke(self, cores=None, niqe=None, scorer=None):
        with redirect_stdout(io.StringIO()):
            return run_experiment(self.plan, cores or self.cores(), lambda image, attack: image,
                                  niqe or (lambda image: 3.0), scorer or Mock(return_value=50),
                                  self.root / 'outputs', seed_fn=Mock())

    def test_same_payload_and_settings_and_unique_outputs(self):
        cores = self.cores()
        directory, code = self.invoke(cores)
        rows = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['payload_sha256'], rows[1]['payload_sha256'])
        for core in cores.values():
            self.assertEqual(core.embed.call_args.kwargs, {'packet_version': 3, 'padding_seed': 73000})
            self.assertEqual(len(core.embed.call_args.args[0]), 8192)
            self.assertEqual(core.extract.call_count, 4)
            for call in core.extract.call_args_list:
                self.assertEqual(set(call.kwargs), {'context'})
        self.assertEqual(code, 0)
        self.assertTrue((directory / 'results.csv').is_file())
        again, _ = self.invoke()
        self.assertNotEqual(directory, again)

    def test_crc_failure_is_counted_and_continues_to_median(self):
        cores = self.cores()
        cores['header_randomized'].extract.side_effect = ['0' * 8192, '0' * 8192,
                                                         ValueError('CRC failure'), '0' * 8192]
        directory, code = self.invoke(cores)
        report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(code, 1)
        self.assertEqual(cores['header_randomized'].extract.call_count, 4)
        self.assertEqual(report['methods']['header_randomized']['extraction_failures']['awgn'], 1)
        self.assertEqual(report['methods']['header_randomized']['mean_acc_including_failures']['awgn'], 0)
        status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['status'], 'complete_with_failures')

    def test_missing_niqe_does_not_produce_candidate_score(self):
        directory, code = self.invoke(niqe=Mock(side_effect=[3.0, float('nan')]))
        report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(code, 1)
        self.assertIsNone(report['methods']['header_randomized']['provisional_score'])
        self.assertIsNone(report['candidate_minus_v3']['score'])

    def test_context_mismatch_is_rejected(self):
        cores = self.cores()
        context = cores['header_randomized'].get_context.return_value
        context['guidance_scale'] = 6.0
        with self.assertRaisesRegex(ValueError, 'contexts differ'):
            self.invoke(cores)
        directory = next((self.root / 'outputs').iterdir())
        status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['status'], 'error')
        self.assertFalse((directory / 'summary.json').exists())

    def test_plan_only_works_without_torch_or_model_files(self):
        with patch.dict('sys.modules', {'torch': None}), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['--plan-only', '--prompts-file', str(self.prompts)]), 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['generation_count'], 10)
        self.assertEqual(plan['inversion_count'], 40)
        self.assertFalse((self.root / 'outputs').exists())
        with self.assertRaises(ValueError):
            load_plan(self.prompts, 100, 73000)

    def test_shared_backend_does_not_initialize_another_model(self):
        with patch('core.steganography.StableDiffusionBackend') as factory:
            normal = SteganographyCore('unused', 'unused', 'unused', device='cpu')
            candidate = RandomizedHeaderCore.from_backend(normal._backend, device=normal.device)
            factory.assert_called_once()
            self.assertIs(candidate._backend, normal._backend)
            self.assertEqual(candidate.device, 'cpu')
            with self.assertRaises(RuntimeError):
                candidate.get_context()
        with self.assertRaises(TypeError):
            RandomizedHeaderCore.from_backend(object())

    def test_recorder_preserves_model_calls_and_arrays(self):
        backend = Mock()
        array = np.ones((1, 4, 64, 64), dtype=np.float32)
        backend.invert.side_effect = [array, RuntimeError('failed inversion'), array]
        recorder = RecordingBackend(backend)
        recorder.prepare(self.root)
        context = {'prompt': 'a cat'}
        recorder.generate(array, context)
        backend.generate.assert_called_once_with(array, context)
        image = Image.new('RGB', (512, 512))
        self.assertIs(recorder.invert(image, context), array)
        with self.assertRaises(RuntimeError):
            recorder.invert(image, context)
        self.assertIs(recorder.invert(image, context), array)
        for name in ('initial_noise.npy', 'inverted_clean.npy', 'inverted_awgn.npy'):
            np.testing.assert_array_equal(np.load(self.root / name, allow_pickle=False), array)
        self.assertFalse((self.root / 'inverted_jpeg.npy').exists())


if __name__ == '__main__':
    unittest.main()
