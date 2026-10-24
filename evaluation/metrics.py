from PIL import Image
import pyiqa
import torch
import numpy as np

# 组员B：初始化 NIQE 评测器，全局加载一次即可
niqe_metric = pyiqa.create_metric('niqe')

def calculate_bpp(total_bits: int, image_size: tuple) -> float:
    """
    计算 bpp (bit per pixel)
    """
    width, height = image_size[0], image_size[1]
    return total_bits / (width * height)

def calculate_acc(original_bits: str, extracted_bits: str) -> float:
    """
    计算比特准确率
    """
    if len(original_bits) != len(extracted_bits):
        # 长度不对齐，直接返回 0
        print("警告：提取比特流长度不匹配！")
        return 0.0
    
    correct = sum(1 for o, e in zip(original_bits, extracted_bits) if o == e)
    return correct / len(original_bits)

def calculate_niqe(image: Image.Image) -> float:
    """
    计算 NIQE，分数越低越自然
    """
    # 组员B注意：pyiqa 默认接收 (B, C, H, W) 的 tensor，范围 [0,1]
    img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)
    
    with torch.no_grad():
        score = niqe_metric(img_tensor).item()
    return score