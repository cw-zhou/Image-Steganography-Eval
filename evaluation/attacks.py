from PIL import Image
import numpy as np
import cv2
import io

def apply_attack(image: Image.Image, attack_type: str) -> Image.Image:
    """
    三种攻击封装
    :param attack_type: 'jpeg', 'awgn', 'mblur'
    """
    # 组员B：直接把 PIL 转成 numpy 数组处理，处理完再转回 PIL
    img_array = np.array(image)

    if attack_type == 'jpeg':
        # Q=70 的 JPEG 压缩攻击
        # 利用内存中的 BytesIO 模拟保存再读取的过程
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        buffer.seek(0)
        return Image.open(buffer)
    
    elif attack_type == 'awgn':
        # 均值 0，标准差 0.05 的高斯噪声
        noise = np.random.normal(0, 0.05, img_array.shape)
        # 假设 img_array 是 0-255，需要将噪声放大到像素尺度，或者先把图归一化到 0-1 加噪声再转回来
        noisy_img = np.clip(img_array + noise * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    elif attack_type == 'mblur':
        # 5x5 中值滤波
        # OpenCV 的 ksize 必须是奇数
        blurred = cv2.medianBlur(img_array, 5)
        return Image.fromarray(blurred)
    
    else:
        # 'identity' (不攻击)
        return image