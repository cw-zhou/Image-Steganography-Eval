import argparse
import os
import torch
from PIL import Image
from core.steganography import SteganographyCore
from evaluation.attacks import apply_attack
from evaluation.metrics import calculate_bpp, calculate_acc, calculate_niqe
from evaluation.scorer import calculate_final_score

def run_pipeline(prompt, secret_bits, core, args):
    print(f"\n>>> 开始处理 Prompt: {prompt}")
    
    # 1. 生成隐写图像，并获取上下文
    stego_image = core.embed(secret_bits, prompt)
    context = core.get_context()  # 🌟 获取提取所需的 context
    
    os.makedirs(args.output_dir, exist_ok=True)
    stego_image.save(os.path.join(args.output_dir, "original_stego.png"))
    
    # 2. 生成攻击图像
    attacks = ['jpeg', 'awgn', 'mblur']
    acc_results = {}
    
    for attack in attacks:
        attacked_image = apply_attack(stego_image, attack)
        attacked_image.save(os.path.join(args.output_dir, f"attacked_{attack}.png"))
        
        # 3. 提取信息（带错误保护，防止一张图失败导致整批崩溃）
        try:
            extracted_bits = core.extract(attacked_image, context=context)  # 🌟 传入 context
        except Exception as e:
            print(f"[{attack.upper()}] 提取报错: {e}")
            extracted_bits = "" # 报错返回空串
            
        # 4. 计算准确率
        acc = calculate_acc(secret_bits, extracted_bits)
        acc_results[attack] = acc
        print(f"[{attack.upper()}] 攻击后提取准确率: {acc:.4f}")
        
    # 5. 计算整体指标
    bpp = calculate_bpp(len(secret_bits), stego_image.size)
    niqe = calculate_niqe(stego_image)
    
    # 6. 计算分数
    score = calculate_final_score(
        bpp=bpp, 
        acc_jpeg=acc_results['jpeg'], 
        acc_awgn=acc_results['awgn'], 
        acc_mblur=acc_results['mblur'], 
        niqe=niqe
    )
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--clip", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    parser.add_argument("--bits", type=str, default="10101010")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    core = SteganographyCore(args.ckpt, args.clip, args.config, device=args.device)
    run_pipeline(args.prompt, args.bits, core, args)