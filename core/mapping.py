"""比特流与 baseline 初始噪声之间的转换；仅依赖 NumPy 和 SciPy。

数据流：消息 -> [版本化长度头 | 消息 | 随机填充] -> (1,4,64,64)
       -> baseline 的正交变换和置乱 -> 初始噪声。
这一步不生成图片；扩散模型的生成和反演由 SteganographyCore 接入。
"""

from typing import Optional, Tuple
import zlib

import numpy as np
from scipy.linalg import qr


class SecretMapping:
    """复用 ours_mapping 的单比特映射，支持单张图容量内的变长消息。

    默认可承载 16384 - 32 = 16352 个原始消息比特。
    bit_length 可限制消息长度上限，兼容 SecretMapping(bit_length=256)。
    解码端只需相同配置和种子，不需要原始消息或上次 encode 的状态。
    种子用于复现映射，不构成消息加密。
    packet_version=1 保留旧格式；版本 2 使用 9 份长度+CRC32，共 576 位，
    默认形状下版本 2 可承载 15808 位消息；候选版本 3 用 31 份头部，容量 14400 位。
    CRC 只校验长度，不校验或纠正消息。padding_seed 仅控制实验填充，解码不需要它。
    """

    HEADER_BITS = 32  # legacy version 1
    HEADER_REPEATS = 9
    PROTECTED_HEADER_BITS = 64 * HEADER_REPEATS
    STRONG_HEADER_REPEATS = 31

    def __init__(
        self,
        bit_length: Optional[int] = None,
        latent_shape: Tuple[int, int, int, int] = (1, 4, 64, 64),
        seed_kernel: int = 100,
        seed_shuffle: int = 101,
        packet_version: int = 1,
        padding_seed: Optional[int] = None,
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
        if type(packet_version) is not int or packet_version not in (1, 2, 3):
            raise ValueError("packet_version 必须为 1、2 或 3")
        self.packet_version = packet_version
        self.header_repeats = self.STRONG_HEADER_REPEATS if packet_version == 3 else self.HEADER_REPEATS
        self.header_bits = self.HEADER_BITS if packet_version == 1 else 64 * self.header_repeats
        if padding_seed is not None and (type(padding_seed) is not int or not 0 <= padding_seed < 2 ** 32):
            raise ValueError("padding_seed 必须是 32 位无符号整数或 None")
        self.padding_seed = padding_seed
        capacity = self.total_bits - self.header_bits
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
        if self.packet_version in (2, 3):
            checksum = zlib.crc32(len(secret_bits).to_bytes(4, "big")) & 0xffffffff
            header = (header + format(checksum, "032b")) * self.header_repeats
        packet = np.frombuffer((header + secret_bits).encode("ascii"), dtype=np.uint8) - 48
        flat = np.random.default_rng(self.padding_seed).integers(0, 2, self.total_bits, dtype=np.uint8)
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
        header = bits[:self.header_bits]
        if self.packet_version in (2, 3):
            # Combine continuous values before thresholding; retain confidence.
            # The packet format is unchanged; CRC and length checks still apply.
            copies = flat[:self.header_bits].reshape(self.header_repeats, 64)
            header = (copies.mean(axis=0) > threshold).astype(np.uint8)
        length = int("".join(str(int(bit)) for bit in header[:32]), 2)
        if self.packet_version in (2, 3):
            received_crc = int("".join(str(int(bit)) for bit in header[32:]), 2)
            expected_crc = zlib.crc32(length.to_bytes(4, "big")) & 0xffffffff
            if received_crc != expected_crc:
                raise ValueError("长度头校验失败：攻击损坏超过当前冗余保护能力")
        if not 1 <= length <= self.max_payload_bits:
            raise ValueError("长度头无效：配置可能不匹配，或消息已被攻击损坏")
        payload = bits[self.header_bits:self.header_bits + length]
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
