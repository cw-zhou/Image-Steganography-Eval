# 本地测试框架使用，可以删除
# 1. 创建 Python 3.8 环境
# conda create -n steg python=3.8 -y
# 2. 激活环境
# conda activate steg
# 3. 安装 PyTorch 2.1.0（CPU 版；如有 GPU 可换成对应 CUDA 索引）
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
# 4. 安装 lmdb
# pip install lmdb==1.4.1
# 5. 安装项目依赖
# pip install -r requirements.txt
# 6. 运行测试
# python test_eval_framework.py

import numpy as np
from PIL import Image
import torch
from evaluation.attacks import apply_attack
from evaluation.metrics import calculate_bpp, calculate_acc, calculate_niqe
from evaluation.scorer import calculate_final_score
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 1. 造一张假图（模拟隐写图）
mock_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

# 2. 测试攻击模块 (组员B)
print("--- 测试攻击模块 ---")
for attack in ['jpeg', 'awgn', 'mblur']:
    attacked_img = apply_attack(mock_image, attack)
    print(f"{attack} 攻击完成，图像尺寸: {attacked_img.size}")

# 3. 测试指标模块 (组员B)
print("\n--- 测试指标模块 ---")
# 测试 bpp
bpp = calculate_bpp(total_bits=256, image_size=(512, 512))
print(f"bpp (256 bits, 512x512): {bpp}") # 期望输出 256 / 262144 ≈ 0.000976

# 测试 Acc
original_bits = "10101010"
extracted_correct = "10101010"
extracted_wrong = "10101011"
print(f"全对准确率: {calculate_acc(original_bits, extracted_correct)}") # 期望 1.0
print(f"错一位准确率: {calculate_acc(original_bits, extracted_wrong)}") # 期望 0.875

# 测试 NIQE
print("正在计算 NIQE (可能需要几秒钟)...")
niqe_score = calculate_niqe(mock_image)
print(f"随机噪点图的 NIQE 分数: {niqe_score:.4f} (通常随机图分数很高，自然图分数低)")

# 4. 测试计分器
print("\n--- 测试计分器 ---")
calculate_final_score(bpp=0.001, acc_jpeg=0.9, acc_awgn=0.8, acc_mblur=0.85, niqe=4.5)