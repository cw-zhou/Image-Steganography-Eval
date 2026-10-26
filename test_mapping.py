"""不需要 GPU、PyTorch 或模型权重的映射验证：python test_mapping.py。"""

import ast
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import qr

from core.mapping import SecretMapping


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.mapping = SecretMapping()

    def test_variable_lengths_and_trailing_zeros(self):
        # 每次使用新的解码实例，验证没有从 encode 缓存原文或消息长度。
        for message in ("0", "1", "010110", "10100000", "01" * 128):
            with self.subTest(length=len(message)):
                noise = self.mapping.encode(message)
                self.assertEqual(noise.shape, (1, 4, 64, 64))
                self.assertEqual(noise.dtype, np.float32)
                self.assertEqual(SecretMapping().decode(noise), message)

    def test_full_capacity(self):
        rng = np.random.default_rng(42)
        message = "".join(str(int(bit)) for bit in rng.integers(0, 2, self.mapping.max_payload_bits))
        self.assertEqual(self.mapping.max_payload_bits, 16352)
        self.assertEqual(SecretMapping().decode(self.mapping.encode(message)), message)

    def test_input_validation_and_capacity(self):
        for message in ("", "0102", " 01", "01\n", "1" * 16353):
            with self.subTest(message=message[:10]):
                with self.assertRaises(ValueError):
                    self.mapping.encode(message)
        with self.assertRaises(TypeError):
            self.mapping.encode([0, 1])
        limited = SecretMapping(bit_length=256)
        self.assertEqual(limited.decode(limited.encode("1" * 256)), "1" * 256)
        with self.assertRaises(ValueError):
            limited.encode("1" * 257)

    def test_threshold_on_recovered_symbols(self):
        symbols = self.mapping.bits_to_tensor("010100")
        # 0 被扰动到 0.1，1 被扰动到 0.9，阈值判决仍应恢复原始消息。
        self.assertEqual(self.mapping.tensor_to_bits(symbols * 0.8 + 0.1), "010100")

    def test_invalid_recovery_is_reported(self):
        with self.assertRaises(ValueError):
            self.mapping.decode(np.zeros((1, 256)))
        with self.assertRaises(ValueError):
            self.mapping.decode(np.full((1, 4, 64, 64), np.nan))
        with self.assertRaises(ValueError):
            self.mapping.tensor_to_bits(np.zeros((1, 4, 64, 64)))
        with self.assertRaises(ValueError):
            self.mapping.tensor_to_bits(self.mapping.bits_to_tensor("01"), threshold=0)

    def test_matches_baseline_math(self):
        # 只载入 baseline 的两个类，跳过无关 torch 导入和失效的示例入口。
        # 不修改 baseline，也不执行其命令行脚本。
        source = Path(__file__).parent / "baseline" / "scripts" / "mapping_module.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ("mapping_module", "ours_mapping")]
        namespace = {"np": np, "qr": qr}
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
        baseline = namespace["ours_mapping"](bits=1)
        noise = self.mapping.encode("001011000")
        state = np.random.get_state()
        try:
            symbols = baseline.decode_secret(noise, seed_kernel=100, seed_shuffle=101)
            expected = baseline.encode_secret(symbols, seed_kernel=100, seed_shuffle=101)
        finally:
            np.random.set_state(state)
        self.assertEqual(self.mapping.tensor_to_bits(symbols), "001011000")
        np.testing.assert_allclose(noise, expected, rtol=1e-6, atol=1e-6)

    def test_does_not_change_global_numpy_random_state(self):
        before = np.random.get_state()
        self.mapping.decode(self.mapping.encode("100100"))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
