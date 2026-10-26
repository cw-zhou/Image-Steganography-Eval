"""比特流与 baseline 初始噪声之间的转换；仅依赖 NumPy 和 SciPy。

数据流：消息 -> [32 位长度头 | 消息 | 随机填充] -> (1,4,64,64)
       -> baseline 的正交变换和置乱 -> 初始噪声。
这一步不生成图片；扩散模型的生成和反演由 SteganographyCore 接入。
"""

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import qr


class SecretMapping:
    """复用 ours_mapping 的单比特映射，支持单张图容量内的变长消息。

    默认可承载 16384 - 32 = 16352 个原始消息比特。
    bit_length 可限制消息长度上限，兼容 SecretMapping(bit_length=256)。
    解码端只需相同配置和种子，不需要原始消息或上次 encode 的状态。
    种子用于复现映射，不构成消息加密。长度头目前没有纠错保护。
    """

    HEADER_BITS = 32

    def __init__(
        self,
        bit_length: Optional[int] = None,
        latent_shape: Tuple[int, int, int, int] = (1, 4, 64, 64),
        seed_kernel: int = 100,
        seed_shuffle: int = 101,
    ):
        if (
            len(latent_shape) != 4
            or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in latent_shape)
            or latent_shape[0] != 1
            or latent_shape[-2] != latent_shape[-1]
        ):
            raise ValueError("latent_shape 必须为 (1, C, H, H)，各维度为正整数")
        self.latent_shape = tuple(latent_shape)
        self.total_bits = int(np.prod(self.latent_shape))
        capacity = self.total_bits - self.HEADER_BITS
        if capacity <= 0 or capacity >= 2 ** self.HEADER_BITS:
            raise ValueError("潜变量容量不足，或消息长度超过长度头可表示的范围")
        if bit_length is not None and (
            not isinstance(bit_length, int)
            or isinstance(bit_length, bool)
            or not 1 <= bit_length <= capacity
        ):
            raise ValueError("bit_length 必须为容量范围内的正整数")
        self.bit_length = capacity if bit_length is None else bit_length
        self.max_payload_bits = self.bit_length

        # 使用局部 RandomState 与 baseline 对齐，避免修改全局随机数状态。
        height = self.latent_shape[-1]
        self._kernel, _ = qr(np.random.RandomState(seed_kernel).randn(height, height))
        self._order = np.arange(self.total_bits)
        np.random.RandomState(seed_shuffle).shuffle(self._order)
        self._reverse_order = np.argsort(self._order)

    def bits_to_tensor(self, secret_bits: str) -> np.ndarray:
        """打包消息，返回 baseline 所需形状的 0/1 NumPy 数组。

        tensor 在这里指多维数组；接模型时再用 torch.from_numpy 转换。
        不截断输入；长度头用于保留消息末尾的 0 和去除填充。
        随机填充避免短消息后面出现大段固定的 0。
        """
        if not isinstance(secret_bits, str):
            raise TypeError("secret_bits 必须是字符串")
        if not secret_bits or any(bit not in "01" for bit in secret_bits):
            raise ValueError("secret_bits 必须是非空的 0/1 字符串")
        if len(secret_bits) > self.max_payload_bits:
            raise ValueError(
                "消息包含 {} bit，当前最多允许 {} bit；不会自动截断".format(
                    len(secret_bits), self.max_payload_bits
                )
            )

        header = format(len(secret_bits), "032b")
        packet = np.frombuffer((header + secret_bits).encode("ascii"), dtype=np.uint8) - 48
        flat = np.random.default_rng().integers(0, 2, self.total_bits, dtype=np.uint8)
        flat[:packet.size] = packet
        return flat.reshape(self.latent_shape).astype(np.float32)

    def _as_array(self, tensor) -> np.ndarray:
        # 也接受模型反演后的 torch.Tensor，无须在本模块导入 torch。
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        array = np.asarray(tensor, dtype=np.float64)
        if array.shape != self.latent_shape:
            raise ValueError("输入形状应为 {}，实际为 {}".format(self.latent_shape, array.shape))
        if not np.isfinite(array).all():
            raise ValueError("输入不能含 NaN 或无穷大")
        return array

    def tensor_to_bits(self, tensor, threshold: float = 0.5) -> str:
        """对解映射后的符号使用阈值，读取长度头并返回真实消息。

        阈值用于已恢复到 0/1 尺度的符号，不能直接用于初始噪声。
        """
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold 必须在 0 和 1 之间")
        flat = self._as_array(tensor).reshape(-1)
        bits = (flat > threshold).astype(np.uint8)
        length = int("".join(str(int(bit)) for bit in bits[:self.HEADER_BITS]), 2)
        if not 1 <= length <= self.max_payload_bits:
            raise ValueError("长度头无效：配置可能不匹配，或消息已被攻击损坏")
        payload = bits[self.HEADER_BITS:self.HEADER_BITS + length]
        return "".join(str(int(bit)) for bit in payload)

    def encode(self, secret_bits: str) -> np.ndarray:
        """消息 -> 初始噪声；与 baseline ours_mapping(bits=1) 的公式一致。"""
        symbols = self.bits_to_tensor(secret_bits)
        standardized = (symbols.astype(np.float64) - 0.5) / 0.5
        noise = self._kernel @ standardized @ self._kernel.T
        noise = noise.reshape(-1)[self._order].reshape(self.latent_shape)
        return noise.astype(np.float32)

    def decode(self, pred_noise, threshold: float = 0.5) -> str:
        """反演噪声 -> 取消置乱和正交变换 -> 阈值判决 -> 原始消息。"""
        noise = self._as_array(pred_noise)
        unshuffled = noise.reshape(-1)[self._reverse_order].reshape(self.latent_shape)
        standardized = self._kernel.T @ unshuffled @ self._kernel
        symbols = standardized * 0.5 + 0.5
        return self.tensor_to_bits(symbols, threshold=threshold)
