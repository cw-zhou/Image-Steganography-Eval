import argparse
import os
import random
import torch
from tqdm import tqdm
from pipeline import run_pipeline
from core.steganography import SteganographyCore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--clip", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True, help="100条prompt的txt")
    parser.add_argument("--output_dir", type=str, default="outputs_batch")
    parser.add_argument("--bits_len", type=int, default=1024, help="测试嵌入的比特数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    print(f"共读取到 {len(prompts)} 条 Prompt，开始加载模型...")
    
    core = SteganographyCore(args.ckpt, args.clip, args.config, device=args.device)
    
    scores = []
    for i, prompt in enumerate(tqdm(prompts, desc="Processing Prompts")):
        # 随机生成二进制比特流 (长度与室友约定好，比如 1024)
        secret_bits = "".join([str(random.randint(0, 1)) for _ in range(args.bits_len)])
        
        # 为每张图分配单独的文件夹，防止覆盖
        args.output_dir = os.path.join("outputs_batch", f"img_{i:03d}")
        
        try:
            score = run_pipeline(prompt, secret_bits, core, args)
            scores.append(score)
        except Exception as e:
            print(f"\n[Error] 第 {i} 张图处理失败，错误: {e}")
            scores.append(0.0) 
            
    print(f"\n========== 批量测试完成 ==========")
    print(f"成功处理 {len(scores)} 条，平均得分: {sum(scores)/len(scores):.2f} / 100")