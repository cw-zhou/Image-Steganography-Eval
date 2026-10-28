"""独立单图验证：只调用 core，不导入 evaluation，不自动下载模型。"""

import argparse
import importlib.metadata
import json
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from core.mapping import SecretMapping
from core.steganography import SteganographyCore


PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_MAX_BITS = 4 * 64 * 64 - SecretMapping.HEADER_BITS
MAX_BITS = 4 * 64 * 64 - SecretMapping.PROTECTED_HEADER_BITS


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("roundtrip", "extract"), default="roundtrip")
    parser.add_argument("--ckpt", required=True, type=Path, help="本地 SD 1.5 .ckpt 文件")
    parser.add_argument("--clip", required=True, type=Path, help="本地 CLIP 权重和 tokenizer 目录")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "baseline/configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", default="cuda", help="例如 cuda、cuda:0、cpu")
    parser.add_argument("--prompt", help="roundtrip 模式的图片描述")
    message = parser.add_mutually_exclusive_group()
    message.add_argument("--bits", help="直接提供 0/1 字符串")
    message.add_argument("--bits-file", type=Path, help="从 UTF-8 文本读取 0/1 字符串")
    message.add_argument("--bit-length", type=int, help="随机测试消息长度；默认 256")
    parser.add_argument("--seed", type=int, default=42, help="仅控制随机测试消息，不保证图片逐像素复现")
    parser.add_argument("--image", type=Path, help="extract 模式的输入图片")
    parser.add_argument("--context", type=Path, help="extract 模式的 context.json")
    parser.add_argument("--reference-bits-file", type=Path, help="extract 模式可选的评分参考，只在提取后用于比较")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/core-smoke")
    return parser


def _validate_bits(bits, max_bits=MAX_BITS):
    if not bits or any(bit not in "01" for bit in bits):
        raise ValueError("秘密信息必须是非空的 0/1 字符串")
    if len(bits) > max_bits:
        raise ValueError("单张图最多支持 {} 个原始消息比特".format(max_bits))
    return bits


def _bits_file(path, max_bits=MAX_BITS):
    # 允许文件首尾换行，不删除消息内部的空白或非法字符。
    return _validate_bits(path.read_text(encoding="utf-8-sig").strip(), max_bits=max_bits)


def _roundtrip_bits(args):
    if args.bits is not None:
        return _validate_bits(args.bits)
    if args.bits_file is not None:
        return _bits_file(args.bits_file)
    length = 256 if args.bit_length is None else args.bit_length
    if not 1 <= length <= MAX_BITS:
        raise ValueError("bit-length 必须在 1～{} 之间".format(MAX_BITS))
    rng = random.Random(args.seed)
    return "".join(str(rng.randrange(2)) for _ in range(length))


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _versions():
    result = {"python": sys.version.split()[0]}
    for package in ("torch", "torchvision", "numpy", "scipy", "Pillow"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _gpu_stats(device):
    # 只使用模型初始化时已经加载的 torch，不为收集信息额外导入或安装它。
    torch = sys.modules.get("torch")
    if torch is None or not str(device).startswith("cuda"):
        return None
    try:
        return {
            "name": torch.cuda.get_device_name(device),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 2),
        }
    except (RuntimeError, ValueError, AssertionError):
        return None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "roundtrip":
        if args.image is not None or args.context is not None or args.reference_bits_file is not None:
            parser.error("image、context、reference-bits-file 仅用于 extract 模式")
        prompt = "A photo of a cat" if args.prompt is None else args.prompt
        if not prompt.strip():
            parser.error("prompt 不能为空")
        try:
            secret = _roundtrip_bits(args)
        except (OSError, UnicodeError, ValueError) as error:
            parser.error(str(error))
    else:
        if args.image is None or args.context is None:
            parser.error("extract 模式必须同时提供 --image 和 --context")
        if any(value is not None for value in (args.bits, args.bits_file, args.bit_length, args.prompt)):
            parser.error("extract 模式从 context 读取 prompt；评分参考请用 --reference-bits-file")
        secret = None

    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir.resolve() / ("run_{}_{}".format(stamp, uuid.uuid4().hex[:8]))
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error("无法创建输出目录：{}".format(error))
    print("Run directory:", run_dir, flush=True)
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "status": "running",
        "stage": "input",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "device": args.device,
        "paths": {name: str(getattr(args, name).resolve()) for name in ("ckpt", "clip", "config")},
        "attack": "none_png_storage" if args.mode == "roundtrip" else "external_image_unspecified",
        "timings_seconds": {},
        "accuracy_policy": "length_mismatch_is_zero",
    }
    exit_code = 2
    try:
        if args.mode == "roundtrip":
            (run_dir / "secret_bits.txt").write_text(secret + "\n", encoding="utf-8")
            report["prompt"] = prompt
            report["message_seed"] = args.seed if args.bits is None and args.bits_file is None else None
        else:
            context = json.loads(args.context.read_text(encoding="utf-8-sig"))
            # 在启动大模型前检查辅助信息的结构。
            context = SteganographyCore._validate_context(context)
            with Image.open(args.image) as opened:
                image = opened.convert("RGB")
            if image.size != (512, 512):
                raise ValueError("输入图片必须为 512×512")
            report["input_image"] = str(args.image.resolve())
            report["prompt"] = context["prompt"]
            _write_json(run_dir / "context.json", context)

        report["stage"] = "load_model"
        checkpoint = time.perf_counter()
        core = SteganographyCore(str(args.ckpt), str(args.clip), str(args.config), device=args.device)
        report["timings_seconds"]["load_model"] = round(time.perf_counter() - checkpoint, 4)
        if args.mode == "roundtrip":
            report["stage"] = "embed"
            checkpoint = time.perf_counter()
            generated = core.embed(secret, prompt)
            report["timings_seconds"]["embed"] = round(time.perf_counter() - checkpoint, 4)
            generated.save(run_dir / "stego.png", format="PNG")
            context = core.get_context()
            _write_json(run_dir / "context.json", context)
            # 实际保存、重读文件，并显式传入 context。
            with Image.open(run_dir / "stego.png") as opened:
                image = opened.convert("RGB")

        report["stage"] = "extract"
        checkpoint = time.perf_counter()
        recovered = core.extract(image, context=context)
        report["timings_seconds"]["extract"] = round(time.perf_counter() - checkpoint, 4)
        if not isinstance(recovered, str) or not recovered or any(bit not in "01" for bit in recovered):
            raise ValueError("提取接口返回的结果不是非空 0/1 字符串")
        (run_dir / "recovered_bits.txt").write_text(recovered + "\n", encoding="utf-8")
        report["recovered_bit_count"] = len(recovered)

        # 评分参考只在提取完成后读取，不传给模型或 extract。
        report["stage"] = "compare"
        if args.mode == "extract" and args.reference_bits_file is not None:
            secret = _bits_file(args.reference_bits_file, max_bits=LEGACY_MAX_BITS)
        if secret is None:
            report.update(status="ok", acc=None, exact_match=None, bpp=None)
            print("Extraction completed; no reference was provided, so accuracy is unknown.", flush=True)
        else:
            length_match = len(secret) == len(recovered)
            correct = sum(left == right for left, right in zip(secret, recovered))
            acc = correct / len(secret) if length_match else 0.0
            exact = secret == recovered
            report.update(
                status="ok" if exact else "mismatch",
                original_bit_count=len(secret),
                length_match=length_match,
                matching_positions=correct,
                acc=acc,
                exact_match=exact,
                bpp=len(secret) / (image.width * image.height),
            )
            print("Acc: {:.6f}; exact match: {}; bpp: {:.8f}".format(acc, exact, report["bpp"]), flush=True)
        report["stage"] = "complete"
        exit_code = 0 if report["status"] == "ok" else 1
    except Exception as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("ERROR at {}: {}".format(report["stage"], error), file=sys.stderr, flush=True)
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        report["gpu"] = _gpu_stats(args.device)
        _write_json(run_dir / "result.json", report)
        print("Result:", run_dir / "result.json", flush=True)
    return exit_code


if __name__ == "__main__":
    # Windows 的终端和工具输出也统一使用 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
