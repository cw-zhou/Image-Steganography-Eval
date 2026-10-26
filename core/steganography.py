"""组员 A 对外提供的隐写接口；模型计算见 diffusion_backend.py。"""

import math
from copy import deepcopy
from typing import Optional

from PIL import Image

from .diffusion_backend import StableDiffusionBackend
from .mapping import SecretMapping


class SteganographyCore:
    """消息 -> 初始噪声 -> 图片，以及图片 -> 反演噪声 -> 消息。

    context 只保存提取所需的 prompt 和参数，不保存原文或初始噪声。
    真实生成需要本地模型文件和完整的 PyTorch/baseline 运行环境。
    """

    DEFAULT_CONTEXT = {
        "version": 2,
        "seed_kernel": 100,
        "seed_shuffle": 101,
        "mapping": "ours_mapping",
        "bits_per_symbol": 1,
        "width": 512,
        "height": 512,
        "guidance_scale": 5.0,
        "generation_steps": 20,
        "inversion_steps": 20,
        "dpm_order": 2,
        "ddim_eta": 0.0,
    }

    def __init__(self, model_path, clip_path, config_path, device="cuda"):
        self.device = str(device)
        self._context = None
        self._backend = StableDiffusionBackend(
            model_path, clip_path, config_path, device=self.device
        )

    @classmethod
    def _validate_context(cls, context: dict) -> dict:
        if not isinstance(context, dict):
            raise TypeError("context 必须是字典")
        expected = set(cls.DEFAULT_CONTEXT) | {"prompt"}
        if set(context) != expected:
            raise ValueError("context 字段不完整或包含未支持的字段")
        result = deepcopy(context)
        if not isinstance(result["prompt"], str) or not result["prompt"].strip():
            raise ValueError("prompt 必须是非空字符串")
        if type(result["version"]) is not int or result["version"] not in (1, 2, 3):
            raise ValueError("不支持的 context version")
        for name, expected_value in (
            ("mapping", "ours_mapping"),
            ("bits_per_symbol", 1), ("width", 512), ("height", 512),
        ):
            if type(result[name]) is not type(expected_value) or result[name] != expected_value:
                raise ValueError("不支持的 context 参数：{}".format(name))
        for name in ("seed_kernel", "seed_shuffle"):
            if type(result[name]) is not int or not 0 <= result[name] < 2 ** 32:
                raise ValueError("{} 必须是 32 位无符号整数".format(name))
        if type(result["dpm_order"]) is not int or result["dpm_order"] not in (1, 2, 3):
            raise ValueError("dpm_order 必须为 1、2 或 3")
        for name in ("generation_steps", "inversion_steps"):
            if type(result[name]) is not int or result[name] < result["dpm_order"]:
                raise ValueError("{} 必须为不小于 dpm_order 的整数".format(name))
        scale = result["guidance_scale"]
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
            raise ValueError("guidance_scale 必须为有限的非负数")
        eta = result["ddim_eta"]
        if type(eta) not in (int, float) or not math.isfinite(eta) or eta != 0:
            raise ValueError("当前仅支持 ddim_eta=0")
        return result

    @staticmethod
    def _mapping(context: dict, *, padding_seed=None) -> SecretMapping:
        return SecretMapping(
            seed_kernel=context["seed_kernel"], seed_shuffle=context["seed_shuffle"],
            packet_version=context["version"], padding_seed=padding_seed,
        )

    def embed(self, secret_bits: str, prompt: str, *, packet_version=None, padding_seed=None) -> Image.Image:
        """以版本 2 嵌入 1～15808 位信息，返回 RGB、512×512 的载密图片。

        只有成功生成图片后才更新最近一次 context。
        可选 packet_version=3 使用 31 份头部、最大 14400 位；默认仍为版本 2。
        padding_seed 只控制实验的随机填充，不作为解码辅助信息。
        """
        context = dict(self.DEFAULT_CONTEXT, prompt=prompt)
        if packet_version is not None:
            context["version"] = packet_version
        context = self._validate_context(context)
        init_noise = self._mapping(context, padding_seed=padding_seed).encode(secret_bits)
        image = self._backend.generate(init_noise, context)
        if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (512, 512):
            raise RuntimeError("模型后端必须返回 RGB、512×512 的 PIL 图片")
        self._context = deepcopy(context)
        return image

    def extract(self, image: Image.Image, *, context: Optional[dict] = None) -> str:
        """从图片反演并恢复信息；可显式传入这张图片对应的 context。"""
        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 PIL.Image.Image")
        if image.size != (512, 512):
            raise ValueError("当前仅支持 512×512 图片，不会自动缩放")
        if context is None:
            context = self.get_context()
        context = self._validate_context(context)
        pred_noise = self._backend.invert(image.convert("RGB"), context)
        # 从图像反演结果解码；没有读取原始消息的捷径。
        return self._mapping(context).decode(pred_noise)

    def get_context(self) -> dict:
        """返回可用 JSON 保存的辅助信息副本。"""
        if self._context is None:
            raise RuntimeError("当前没有提取上下文，请先 embed，或给 extract 显式传入 context")
        return deepcopy(self._context)
