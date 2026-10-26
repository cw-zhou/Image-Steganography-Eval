"""验证单图脚本的文件和错误处理；测试替身不代表真实模型效果。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import run_core_smoke
from core.steganography import SteganographyCore


class SmokeScriptTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parent / "outputs" / "test-smoke"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(test_root))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def args(self, *extra):
        return ["--ckpt", "unused.ckpt", "--clip", "unused_clip", "--device", "cpu",
                "--output-dir", str(self.root), *extra]

    def invoke(self, arguments):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_core_smoke.main(arguments)

    def result(self):
        reports = list(self.root.glob("run_*/result.json"))
        self.assertEqual(len(reports), 1)
        return reports[0].parent, json.loads(reports[0].read_text(encoding="utf-8"))

    def test_v2_capacity_checked_before_model_load(self):
        for option, value in (("--bits", "1" * 15809), ("--bit-length", "15809")):
            with patch.object(run_core_smoke, "SteganographyCore") as factory:
                with self.assertRaises(SystemExit) as caught:
                    self.invoke(self.args(option, value))
                self.assertEqual(caught.exception.code, 2)
                factory.assert_not_called()
        self.assertEqual(list(self.root.glob("run_*")), [])
        self.assertEqual(len(run_core_smoke._validate_bits("1" * 16352, max_bits=run_core_smoke.LEGACY_MAX_BITS)), 16352)

    def test_roundtrip_reopens_png_and_saves_all_outputs(self):
        with patch.object(run_core_smoke, "SteganographyCore") as factory:
            core = factory.return_value
            core.embed.return_value = Image.new("RGB", (512, 512), (17, 30, 90))
            context = dict(SteganographyCore.DEFAULT_CONTEXT, prompt="a cat")
            core.get_context.return_value = context
            core.extract.return_value = "010100"
            code = self.invoke(self.args("--bits", "010100", "--prompt", "a cat"))
            image = core.extract.call_args.args[0]
            self.assertIsNot(image, core.embed.return_value)
            self.assertEqual(image.getpixel((0, 0)), (17, 30, 90))
            self.assertEqual(core.extract.call_args.kwargs, {"context": context})
        directory, report = self.result()
        self.assertEqual(code, 0)
        self.assertEqual(report["acc"], 1.0)
        self.assertAlmostEqual(report["bpp"], 6 / (512 * 512))
        self.assertEqual(report["attack"], "none_png_storage")
        for name in ("stego.png", "secret_bits.txt", "recovered_bits.txt", "context.json"):
            self.assertTrue((directory / name).is_file())

    def test_mismatch_and_length_mismatch_do_not_report_success(self):
        for recovered, expected_acc in (("010101", 5 / 6), ("0101000", 0.0)):
            with self.subTest(recovered=recovered):
                output = self.root / recovered
                with patch.object(run_core_smoke, "SteganographyCore") as factory:
                    core = factory.return_value
                    core.embed.return_value = Image.new("RGB", (512, 512))
                    core.get_context.return_value = dict(SteganographyCore.DEFAULT_CONTEXT, prompt="a cat")
                    core.extract.return_value = recovered
                    arguments = self.args("--bits", "010100", "--output-dir", str(output))
                    code = self.invoke(arguments)
                path = next(output.glob("run_*/result.json"))
                report = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "mismatch")
                self.assertAlmostEqual(report["acc"], expected_acc)

    def test_missing_weights_produce_error_report_with_traceback(self):
        code = self.invoke(self.args("--bits", "01"))
        directory, report = self.result()
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["stage"], "load_model")
        self.assertEqual(report["error"]["type"], "FileNotFoundError")
        self.assertTrue((directory / "traceback.txt").is_file())

    def test_extract_only_passes_context_and_can_skip_accuracy(self):
        image = self.root / "input.png"
        context_path = self.root / "input_context.json"
        Image.new("RGB", (512, 512)).save(image)
        context = dict(SteganographyCore.DEFAULT_CONTEXT, prompt="a cat")
        context_path.write_text(json.dumps(context), encoding="utf-8")
        with patch.object(run_core_smoke, "SteganographyCore") as factory:
            # 保留真实 context 校验，测试替身仅代替模型实例。
            factory._validate_context.side_effect = SteganographyCore._validate_context
            core = factory.return_value
            core.extract.return_value = "101"
            code = self.invoke(self.args("--mode", "extract", "--image", str(image), "--context", str(context_path)))
            core.embed.assert_not_called()
            self.assertEqual(core.extract.call_args.kwargs, {"context": context})
        _, report = self.result()
        self.assertEqual(code, 0)
        self.assertIsNone(report["acc"])
        self.assertIsNone(report["bpp"])

    def test_invalid_message_does_not_initialize_model_or_write_results(self):
        with patch.object(run_core_smoke, "SteganographyCore") as factory:
            with self.assertRaises(SystemExit) as raised:
                self.invoke(self.args("--bits", "0102"))
            factory.assert_not_called()
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(list(self.root.glob("run_*")), [])

    def test_reference_is_read_only_after_extraction(self):
        image = self.root / "input.png"
        context_path = self.root / "input_context.json"
        reference = self.root / "reference.txt"
        Image.new("RGB", (512, 512)).save(image)
        context = dict(SteganographyCore.DEFAULT_CONTEXT, prompt="a cat")
        context_path.write_text(json.dumps(context), encoding="utf-8")

        def extraction_finished(*args, **kwargs):
            # 参考文件到提取结束才出现；如果提前读取，此测试会失败。
            reference.write_text("101000\n", encoding="utf-8")
            return "101000"

        with patch.object(run_core_smoke, "SteganographyCore") as factory:
            factory._validate_context.side_effect = SteganographyCore._validate_context
            factory.return_value.extract.side_effect = extraction_finished
            code = self.invoke(self.args(
                "--mode", "extract", "--image", str(image), "--context", str(context_path),
                "--reference-bits-file", str(reference),
            ))
        _, report = self.result()
        self.assertEqual(code, 0)
        self.assertEqual(report["acc"], 1.0)
        self.assertEqual(report["original_bit_count"], 6)

    def test_unique_directories_preserve_previous_results(self):
        with patch.object(run_core_smoke, "SteganographyCore") as factory:
            core = factory.return_value
            core.embed.return_value = Image.new("RGB", (512, 512))
            core.get_context.return_value = dict(SteganographyCore.DEFAULT_CONTEXT, prompt="a cat")
            core.extract.return_value = "01"
            self.assertEqual(self.invoke(self.args("--bits", "01")), 0)
            self.assertEqual(self.invoke(self.args("--bits", "01")), 0)
        self.assertEqual(len(list(self.root.glob("run_*/result.json"))), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
