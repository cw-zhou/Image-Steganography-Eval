from PIL import Image
import numpy as np
import cv2
import io

def apply_attack(image: Image.Image, attack_type: str) -> Image.Image:
    # 统一转 RGB 防呆
    image = image.convert("RGB")
    img_array = np.array(image)

    if attack_type == 'jpeg':
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        buffer.seek(0)
        # 用 copy() 读取，防止内存泄漏
        return Image.open(buffer).convert("RGB").copy()
    
    elif attack_type == 'awgn':
        noise = np.random.normal(0, 0.05, img_array.shape)
        noisy_img = np.clip(img_array + noise * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    elif attack_type == 'mblur':
        blurred = cv2.medianBlur(img_array, 5)
        return Image.fromarray(blurred)
    
    else:
        return image
