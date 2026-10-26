# 组员 A：嵌入与提取接口约定

状态：接口封装已实现。`core/mapping.py` 已通过 7 项 CPU 映射测试；
`core/steganography.py` 和 `core/diffusion_backend.py` 已实现真实模型调用路径，
`get_context()` 和 `extract(..., context=...)` 已实现。
另外 12 项本地测试覆盖接口衔接、输入校验、上下文、采样调用方向和真实图像数据转换；
其中模型计算使用测试替身，真实图像生成、信息提取和抗攻击效果仍待 GPU 验证。

## 1. 对外接口

保留 README 中同学已经使用的类名、构造参数和基本调用方式：

```python
class SteganographyCore:
    def __init__(self, model_path, clip_path, config_path, device="cuda"):
        ...

    def embed(self, secret_bits: str, prompt: str) -> Image.Image:
        ...

    def extract(self, image: Image.Image, *, context=None) -> str:
        ...

    def get_context(self) -> dict:
        ...
```

| 方法 | 输入 | 输出和职责 |
|---|---|---|
| `__init__` | SD、CLIP、配置路径和运行设备 | 加载一次模型，初始化采样器和映射器 |
| `embed` | 非空的 0/1 字符串、非空 prompt | 返回一张 RGB、512×512 的 `PIL.Image.Image` |
| `extract` | 原图或攻击图，可选的提取上下文 | 返回从图像计算得到的 0/1 字符串 |
| `get_context` | 无 | 返回最近一次成功嵌入所需的辅助信息副本，可保存为 JSON |

默认采用单比特映射、潜变量形状 `(1,4,64,64)`。
32 位长度头加上消息和随机填充共占 16384 位，因此当前原始消息长度范围为 1～16352 位。
超长或含非 0/1 字符时明确报错，不截断。真实消息长度从解码出的长度头读取。
当前长度头没有纠错保护，抗攻击效果待 GPU 实验检验。

## 2. 提取端需要的辅助信息

baseline 反演用到生成时的 prompt，解映射用到相同的映射种子。
context 约定包含以下字段：

```python
{
    "version": 1,
    "prompt": "a photo of a cat",
    "seed_kernel": 100,
    "seed_shuffle": 101,
    "mapping": "ours_mapping",
    "bits_per_symbol": 1,
    "width": 512,
    "height": 512,
    "guidance_scale": 5.0,
    "generation_steps": 20,
    "inversion_steps": 20,
    "dpm_order": 2,
    "ddim_eta": 0.0,
}
```

第一版采用固定的映射种子；这用于复现映射，不提供秘密信息加密。
context 不包含原始秘密比特、初始噪声或正确答案。
提取端必须使用相同的模型权重和模型配置；context 不会替代模型文件。
这些是 baseline 所需的共享辅助信息，不应把当前方案描述为“只凭图片、无需任何上下文”提取。

## 3. 与现有评测代码衔接

同一个实例在 `embed` 成功后记住本次 context，因此现有调用仍然有效：

```python
image = core.embed("010110", "a photo of a cat")
attacked = apply_attack(image, "jpeg")
recovered = core.extract(attacked)
```

如果要保存图片以后提取，或者切换到另一个模型实例，需要保存并显式传入 context：

```python
image = core.embed("010110", "a photo of a cat")
context = core.get_context()
# 图片和 context 分别保存；另一个实例加载相同的模型权重和配置。
recovered = another_core.extract(image, context=context)
```

每次成功 `embed` 都会更新最近一次 context。处理旧图片或交错处理多张图片时，
必须传入该图片对应的 context，不能默认使用最后一张图的 prompt。
显式传入的 context 优先，未提供时才使用当前实例最近一次成功嵌入的 context。
无可用 context、图片尺寸不匹配或解码长度头非法时明确报错，不返回固定比特或原始消息。

## 4. 内部代码如何分配

| baseline 中的操作 | 封装位置 |
|---|---|
| `load_model_from_config`、配置和采样器加载 | `__init__` |
| prompt 编码、消息映射、正向采样、图像解码 | `embed` |
| 图像编码、逆向采样、噪声解映射、长度头解析 | `extract` |
| 攻击操作、Acc、bpp、NIQE 和总分 | 组员 B 的评测模块 |

`embed` 的图像张量需要从 `[-1,1]` 转成 PIL 像素；`extract` 先做相反转换。
`extract` 的阈值判决发生在解映射后的 0/1 符号尺度，不能直接对初始噪声按 0.5 判决。
原始比特只由调用方提供给评测模块，用来与提取结果比较。

模型加载、正向生成和逆向反演集中在 `core/diffusion_backend.py`。
模型依赖在后端初始化时才导入，因此无 PyTorch 的本地环境仍可运行接口测试：

```powershell
.\.venv\Scripts\python.exe test_mapping.py
.\.venv\Scripts\python.exe test_core.py
```

下一步：在 GPU 服务器准备本地 SD 1.5、CLIP 权重及 `requirements-core.txt` 中的候选依赖，
先验证单张图的无攻击嵌入和提取，再接入三种攻击。候选依赖尚未在真实模型环境完整验证。

## 5. 单图运行入口

`run_core_smoke.py` 提供单图生成/保存/提取，以及另一进程只凭图片和 context 提取的入口。
它独立于组员 B 的评测模块，输出准确率、bpp、版本、耗时和错误报告。
具体环境、权重准备、命令和结果解释见 [服务器运行说明](SERVER_RUNBOOK.md)。
另外 7 项 `test_core_smoke.py` 测试覆盖脚本文件处理和异常记录，模型部分同样使用替身。
