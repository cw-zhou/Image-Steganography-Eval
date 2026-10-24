import torch
import numpy as np

class SecretMapping:
    def __init__(self, bit_length=256):
        """
        :param bit_length: 一次固定嵌入的比特数，建议固定下来（如 256 bits）
        """
        self.bit_length = bit_length

    def bits_to_tensor(self, secret_bits: str) -> torch.Tensor:
        """
        【嵌入前处理】将 0/1 字符串转换为 Baseline 需要的张量格式
        组员A：你需要根据 Baseline 源码中 mapping_func 的要求来改这里。
                比如 Baseline 可能要求传入形状为 (1, 256) 的 float 张量。
        """
        # 1. 长度校验与补齐 (Padding)
        if len(secret_bits) < self.bit_length:
            secret_bits = secret_bits + '0' * (self.bit_length - len(secret_bits))
        elif len(secret_bits) > self.bit_length:
            secret_bits = secret_bits[:self.bit_length] # 截断，或者报错

        # 2. 字符串转 Tensor (例如 "101" -> [1.0, 0.0, 1.0])
        bit_list = [float(bit) for bit in secret_bits]
        tensor = torch.tensor(bit_list, dtype=torch.float32)
        
        # 3. 形状调整 (根据 baseline 调整，这里假设需要 (1, 256))
        return tensor.unsqueeze(0)

    def tensor_to_bits(self, tensor: torch.Tensor, threshold=0.5) -> str:
        """
        【提取后处理】将 Baseline 提取出的浮点特征张量转换为 0/1 字符串
        组员A：这里是保证准确率的关键，可能需要引入纠错码(ECC)或调整阈值。
        """
        # 展平并脱离计算图
        flat_tensor = tensor.flatten().detach().cpu().numpy()
        
        # 根据阈值二值化：大于 0.5 视作 1，否则为 0
        bits = "".join(['1' if val > threshold else '0' for val in flat_tensor])
        
        # 如果有 Padding，这里可以截断回实际长度（如果你在包头加了长度信息，在这里处理）
        return bits[:self.bit_length]