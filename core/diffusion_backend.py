"""从 baseline 整理的模型加载、正向生成和逆向反演。

导入此模块不需要 PyTorch；创建后端实例时才加载模型依赖。
只接受本地权重路径，不自动下载模型。
"""

import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image


def image_to_array(image: Image.Image) -> np.ndarray:
    """RGB 图片 -> (1,3,512,512)、[-1,1] 的 float32 数组。"""
    if not isinstance(image, Image.Image) or image.size != (512, 512):
        raise ValueError("输入必须是 512×512 的 PIL 图片")
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    return (pixels / 127.5 - 1.0).transpose(2, 0, 1)[None].copy()


def array_to_image(samples) -> Image.Image:
    """模型输出 -> 裁剪并量化为 RGB 图片。"""
    array = np.asarray(samples, dtype=np.float32)
    if array.shape != (1, 3, 512, 512) or not np.isfinite(array).all():
        raise ValueError("模型图片输出必须是有限的 (1,3,512,512) 数组")
    pixels = np.rint((array[0].transpose(1, 2, 0).clip(-1, 1) + 1.0) * 127.5)
    return Image.fromarray(pixels.astype(np.uint8))


class StableDiffusionBackend:
    def __init__(self, model_path, clip_path, config_path, device="cuda"):
        # 提前检查路径。没有模型文件时直接报错，不触发在线下载。
        model_path = self._local_path(model_path, "SD 权重", directory=False)
        clip_path = self._local_path(clip_path, "CLIP 权重目录", directory=True)
        config_path = self._local_path(config_path, "模型配置", directory=False)
        project_root = Path(__file__).resolve().parents[1]
        self._configure_cache(project_root)
        try:
            import torch
            from omegaconf import OmegaConf
        except ImportError as error:
            raise RuntimeError(
                "模型运行依赖尚未安装；请在 GPU 服务器准备 requirements-core.txt 中的环境"
            ) from error
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前 Python 环境无法使用 CUDA，请检查 GPU 版 PyTorch 和驱动")

        # baseline 使用绝对导入 ldm；按本文件位置定位，避免依赖终端工作目录。
        baseline_root = project_root / "baseline"
        if not (baseline_root / "ldm").is_dir():
            raise FileNotFoundError("未找到项目 baseline/ldm 目录")
        if str(baseline_root) not in sys.path:
            sys.path.insert(0, str(baseline_root))
        try:
            import ldm
            if baseline_root not in Path(ldm.__file__).resolve().parents:
                raise RuntimeError("当前 Python 已加载其他位置的 ldm，请在独立进程运行此项目")
            from ldm.util import instantiate_from_config
            from ldm.models.diffusion.dpm_solver import DPMSolverSampler
        except ImportError as error:
            raise RuntimeError("baseline 依赖不完整，请检查 requirements-core.txt") from error

        config = OmegaConf.load(str(config_path))
        params = config.model.params
        if params.channels != 4:
            raise ValueError("当前映射封装要求模型潜变量通道数为 4")
        # 在内存中覆盖 CLIP 的机器路径，原始 yaml 保持不变。
        params.cond_stage_config.params.version = str(clip_path)
        params.cond_stage_config.params.device = str(self.device)
        model = instantiate_from_config(config.model)
        checkpoint = torch.load(str(model_path), map_location="cpu")
        weights = checkpoint.get("state_dict", checkpoint)
        incompatible = model.load_state_dict(weights, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logging.getLogger(__name__).warning(
                "权重加载：缺失 %d 项，多余 %d 项；请核对 SD 1.5 配置和权重",
                len(incompatible.missing_keys), len(incompatible.unexpected_keys),
            )
        del weights, checkpoint
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.sampler = DPMSolverSampler(self.model)

    @staticmethod
    def _local_path(value, label, directory):
        if value is None or not str(value).strip():
            raise ValueError("{}路径不能为空".format(label))
        path = Path(value).expanduser().resolve()
        if not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError("未找到{}：{}".format(label, path))
        return path

    @staticmethod
    def _configure_cache(project_root):
        # 此项目在 D 盘；服务器上则使用服务器项目目录下的 weights/cache。
        cache = project_root / "weights" / "cache"
        paths = {
            "HF_HOME": cache / "huggingface",
            "HF_HUB_CACHE": cache / "huggingface" / "hub",
            "TORCH_HOME": cache / "torch",
            "MPLCONFIGDIR": cache / "matplotlib",
        }
        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)

    def _precision_scope(self):
        # 对齐 baseline：CUDA 使用 autocast，CPU 使用完整精度。
        if self.device.type == "cuda":
            return self.torch.autocast(device_type="cuda")
        return nullcontext()

    def _conditioning(self, context):
        c = self.model.get_learned_conditioning([context["prompt"]])
        uc = self.model.get_learned_conditioning([""]) if context["guidance_scale"] != 1.0 else None
        return c, uc

    @staticmethod
    def _sampling_args(context, c, uc):
        return {
            "conditioning": c,
            "unconditional_conditioning": uc,
            "batch_size": 1,
            "shape": (4, 64, 64),
            "verbose": False,
            "unconditional_guidance_scale": context["guidance_scale"],
            "eta": context["ddim_eta"],
            "order": context["dpm_order"],
            "width": context["width"],
            "height": context["height"],
        }

    def generate(self, init_noise: np.ndarray, context: dict) -> Image.Image:
        torch = self.torch
        with torch.no_grad(), self._precision_scope():
            c, uc = self._conditioning(context)
            initial = torch.from_numpy(np.asarray(init_noise, dtype=np.float32)).to(self.device)
            z_0, _ = self.sampler.sample(
                steps=context["generation_steps"],
                x_T=initial, DPMencode=False, DPMdecode=True,
                **self._sampling_args(context, c, uc)
            )
            samples = self.model.decode_first_stage(z_0)
        return array_to_image(samples.detach().float().cpu().numpy())

    def invert(self, image: Image.Image, context: dict) -> np.ndarray:
        torch = self.torch
        samples = torch.from_numpy(image_to_array(image)).to(self.device)
        with torch.no_grad(), self._precision_scope():
            c, uc = self._conditioning(context)
            encoded = self.model.encode_first_stage(samples)
            latent = self.model.get_first_stage_encoding(encoded)
            noise, _ = self.sampler.sample(
                steps=context["inversion_steps"],
                x_T=latent, DPMencode=True,
                **self._sampling_args(context, c, uc)
            )
        return noise.detach().float().cpu().numpy()
