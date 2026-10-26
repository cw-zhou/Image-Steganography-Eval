# 本地 GPU 环境

本地开发使用项目自己的 `.venv`。目前本机为 Windows、Python 3.10、RTX 4060 Laptop 8 GB。
最终仍需在老师指定的 Ubuntu、Python 3.8、PyTorch 2.1.0 环境验证。

## 1. 进入项目并准备环境

```powershell
cd "D:\NIS4347\Image-Steganography-Eval"
# 仅在 .venv 不存在时执行：
python -m venv .venv
```

直接使用 `.venv` 中的 Python，无需先运行 PowerShell 激活脚本。

## 下载、临时文件和模型缓存放 D 盘

在当前终端设置以下环境变量，供本次终端中的安装和模型运行使用：

```powershell
$env:PIP_CACHE_DIR = "D:\NIS4347\Image-Steganography-Eval\.venv\pip-cache"
$env:TEMP = "D:\NIS4347\Image-Steganography-Eval\.venv\tmp"
$env:TMP = $env:TEMP
$env:HF_HOME = "D:\NIS4347\Image-Steganography-Eval\weights\cache\huggingface"
$env:HF_HUB_CACHE = "$env:HF_HOME\hub"
$env:TORCH_HOME = "D:\NIS4347\Image-Steganography-Eval\weights\cache\torch"
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR,$env:TEMP,$env:HF_HOME,$env:TORCH_HOME | Out-Null
```

本项目 `.venv/pip.ini` 也已设置 `cache-dir`，所以重新打开终端后，使用这个
虚拟环境的 pip 仍会把缓存放 D 盘。上面的其他环境变量需在新终端重新设置。
`.venv/` 和 `weights/` 已被 Git 忽略。后续 SD、CLIP 权重也放在 `weights/` 下。

## 2. 安装计算库和 GPU 版 PyTorch

```powershell
.\.venv\Scripts\python.exe -m pip install numpy==1.24.4 scipy==1.10.1
.\.venv\Scripts\python.exe -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

选择 NumPy 1.x 以配合较早的 PyTorch；这两个计算库版本也支持 Python 3.8。
CUDA 12.1 指 PyTorch 安装包采用的运行时版本，不要求 `nvidia-smi` 显示的版本与之逐字一致。
此步骤使用已安装的 NVIDIA 驱动，不需要先为运行 PyTorch 安装完整 CUDA Toolkit。

官方安装组合：https://pytorch.org/get-started/previous-versions/#v210

## 3. 验证

```powershell
.\.venv\Scripts\python.exe check_gpu.py
.\.venv\Scripts\python.exe test_mapping.py
.\.venv\Scripts\python.exe test_core.py
.\.venv\Scripts\python.exe -m pip check
```

`check_gpu.py` 会检查软件版本、GPU 可用性、实际的 FP32/FP16 矩阵计算，
以及 NumPy 和 GPU 张量之间的数据转换。失败时返回非零退出码。

`test_mapping.py` 验证消息转换、长度边界和 baseline 映射公式；不加载生成模型。

`test_core.py` 无需 PyTorch，使用模型测试替身验证接口衔接，并实际测试图片数值转换。
它不能证明真实模型已完成嵌入和提取。

核心封装已经实现。GPU 运行环境准备好后，补齐 `requirements-core.txt` 中的候选依赖、
准备 SD 1.5 和 CLIP 本地权重，再验证真实图片生成和反演。
当前的 `requirements.txt` 尚未完整约束 baseline 依赖；本步骤不把它作为已验证的完整环境。

8 GB 显存能否运行整个隐写流程，需要加载模型后测量；小矩阵验证不代表模型已经跑通。
