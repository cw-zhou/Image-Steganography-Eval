# 生成式图像隐写大作业 - 协作开发规范

## 1. 整体架构说明
本项目采用分层架构：`baseline/` 作为底层生成模型，`core/` 负责隐写，`evaluation/` 负责测试，`pipeline.py` 负责串联。

```powershell
Steganography/
│
├── README.md                     # 整体开发规范与协作说明
├── requirements.txt              # 依赖库 (torch, pyiqa, scikit-image, opencv-python, etc.)
├── .gitignore                    # 忽略大文件 (如 *.ckpt, *.pth, outputs/)
│
├── weights/                      # (本地建，不上传) 放 SD1.5 和 CLIP 权重
│
├── baseline/                     # (原封不动放 mas_GRDH 的代码)
│   ├── ldm/
│   ├── scripts/
│   └── configs/
│
├── core/                         # 组员 A (隐写算法)
│   ├── __init__.py
│   ├── steganography.py          # 封装 Baseline 的嵌入/提取逻辑
│   └── mapping.py                # 比特流 <-> 张量的映射/解映射逻辑
│
├── evaluation/                   # 组员 B (攻击与测试)
│   ├── __init__.py
│   ├── attacks.py                # 三种攻击的封装 (JPEG, AWGN, Median)
│   ├── metrics.py                # NIQE, bpp, Acc 的计算逻辑
│   └── scorer.py                 # 按照PPT公式计算最终得分
│
├── pipeline.py                   # 主线流程 (框架制定者B负责维护)
├── run_single_test.py            # 单条 Prompt 跑通全流程测试脚本
└── run_batch_test.py             # 100条 Prompt 批量测试脚本 (租用3090后跑)
```

## 2. 接口标准

### 2.1 组员 A (隐写算法) 提供的接口
在 `core/steganography.py` 中必须实现以下类：
```python
class SteganographyCore:
    def __init__(self, model_path, clip_path, config_path):
        # 加载模型
        pass

    def embed(self, secret_bits: str, prompt: str) -> PIL.Image:
        """
        嵌入算法
        :param secret_bits: 原始二进制字符串，如 "0101101..."
        :param prompt: 文本提示词
        :return: 生成的隐写图像 (PIL.Image格式)
        """
        pass

    def extract(self, image: PIL.Image) -> str:
        """
        提取算法
        :param image: 可能是原图，也可能是经过攻击的图
        :return: 提取出的二进制字符串
        """
        pass
```

### 2.2 组员 B (测试与评估) 提供的接口
在 `evaluation/attacks.py`、`metrics.py`、`scorer.py` 中必须实现：

攻击接口 (`attacks.py`)：
```python
def apply_attack(image: PIL.Image, attack_type: str) -> PIL.Image:
    """
    :param attack_type: 'jpeg', 'awgn', 'mblur'
    """
    pass
```

指标接口 (`metrics.py`)：
```python
def calculate_bpp(total_bits: int, image_size: tuple) -> float:
    # 秘密信息总比特数 / (宽*高)
    pass

def calculate_acc(original_bits: str, extracted_bits: str) -> float:
    # 正确提取的比特数 / 隐藏的总比特数
    pass

def calculate_niqe(image: PIL.Image) -> float:
    # 计算 NIQE 指标
    pass
```

得分计算接口 (`scorer.py`)：
```python
def calculate_final_score(bpp: float, acc_jpeg: float, acc_awgn: float, acc_mblur: float, niqe: float) -> float:
    # 严格按照PPT里的公式进行归一化计算
    pass
```

## 3. 开发任务分工
组员 A (嵌入与提取)：

- 阅读 `baseline/scripts/txt2img.py`，剥离出不相关的代码。
- 修改 mapping_func，实现任意长度 0/1 序列与潜在空间的映射。
- 完善 `core/steganography.py`，跑通单张图的“嵌入->提取”。
- 注意：提取时需要使用阈值将特征恢复成 0/1 比特。

组员 B (攻击与评估)：
- 实现 `evaluation/attacks.py`，直接调用 torchvision 或 cv2 实现 PPT 要求的三种攻击（注意标准差、质量因子等参数）。
- 引入 pyiqa 库实现 NIQE 计算。注意 NIQE 需要在 ycbcr 颜色空间或转灰度图计算。
- 完善 `scorer.py`，注意 PPT 里的公式细节。
- 维护 pipeline.py 和 run_batch_test.py。

## 4. 主线流程说明 (pipeline.py)
主线流程必须是一个闭环，逻辑如下：

- 读入 Prompt 和随机生成的 secret_bits。
- 调用 core.embed() 生成隐写图。
- 保存原图，并调用 evaluation.apply_attac() 生成三张攻击图（JPEG, AWGN, MBLUR）。
- 调用 core.extract() 对原图和三张攻击图进行提取。
- 调用 evaluation.metrics 计算 bpp、各个 Acc 和 NIQE。
- 调用 evaluation.scorer 计算总分，并打印结果。

## “对齐”小贴士：

1. Baseline的隐写是基于扩散模型潜空间的扰动。由于 `embed` 时要保证生成图片的 NIQE 不太高，A 需要研究 Baseline 里面是如何控制扰动强度的。另外，输入任意长度比特时，长度不够如何 Padding？长度超了怎么办？这要在A的代码里写死规则（比如固定嵌入256 bits）。
2. **关于路径**：在 `pipeline.py` 中，不要把路径写死，使用 `argparse` 传入权重路径和配置文件路径，这样方便你（无GPU）和同学（有3060）在不同环境下运行。最后在租的服务器上也能直接跑。
