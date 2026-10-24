import argparse
import os
import torch
from PIL import Image

# 导入你们自己写的模块
from core.steganography import SteganographyCore
from evaluation.attacks import apply_attack
from evaluation.metrics import calculate_bpp, calculate_acc, calculate_niqe
from evaluation.scorer import calculate_final_score

def main():
    parser = argparse.ArgumentParser(description="单条Prompt全流程测试")
    parser.add_argument("--ckpt", type=str, required=True, help="SD 1.5 权重路径 (.ckpt)")
    parser.add_argument("--clip", type=str, required=True, help="CLIP 权重路径")
    parser.add_argument("--config", type=str, required=True, help="Baseline ldm.yaml 路径")
    parser.add_argument("--prompt", type=str, default="A highly detailed portrait of a cyberpunk girl", help="测试 Prompt")
    parser.add_argument("--bits", type=str, default="1010101100110011", help="待嵌入的二进制字符串")
    parser.add_argument("--output_dir", type=str, default="outputs_single", help="输出文件夹")
    args = parser.parse_args()

    # 创建设备和输出目录
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"当前运行设备: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 实例化核心隐写类
    print("\n[1/5] 初始化隐写算法...")
    core = SteganographyCore(args.ckpt, args.clip, args.config, device=device)

    # 2. 嵌入信息并生成图像
    print(f"[2/5] 开始生成图像并嵌入信息 (Bits: {args.bits})...")
    stego_image = core.embed(args.bits, args.prompt)
    original_path = os.path.join(args.output_dir, "stego_original.png")
    stego_image.save(original_path)
    print(f"  -> 隐写图像已保存至: {original_path}")

    # 3. 无攻击情况下的提取 (验证 Embed/Extract 闭环)
    print("[3/5] 测试无攻击提取...")
    extracted_bits_clean = core.extract(stego_image)
    acc_clean = calculate_acc(args.bits, extracted_bits_clean)
    print(f"  -> 无攻击提取准确率: {acc_clean * 100:.2f}%")

    # 4. 三种攻击测试
    print("[4/5] 测试三种攻击场景...")
    attacks = ['jpeg', 'awgn', 'mblur']
    acc_results = {}
    
    for attack in attacks:
        # 施加攻击
        attacked_image = apply_attack(stego_image, attack)
        attacked_path = os.path.join(args.output_dir, f"stego_{attack}.png")
        attacked_image.save(attacked_path)
        
        # 提取信息
        extracted_bits = core.extract(attacked_image)
        acc = calculate_acc(args.bits, extracted_bits)
        acc_results[attack] = acc
        
        print(f"  -> [{attack.upper()}] 攻击准确率: {acc * 100:.2f}% (图已存至 {attacked_path})")

    # 5. 计算指标和得分
    print("[5/5] 计算评价指标...")
    bpp = calculate_bpp(len(args.bits), stego_image.size)
    niqe = calculate_niqe(stego_image)
    
    print(f"  -> 嵌入容量 bpp: {bpp:.6f}")
    print(f"  -> 图像自然度 NIQE: {niqe:.4f}")
    
    # 调用计分器
    total_score = calculate_final_score(
        bpp=bpp,
        acc_jpeg=acc_results['jpeg'],
        acc_awgn=acc_results['awgn'],
        acc_mblur=acc_results['mblur'],
        niqe=niqe
    )
    
    print("\n========== 测试完成 ==========")

if __name__ == "__main__":
    main()