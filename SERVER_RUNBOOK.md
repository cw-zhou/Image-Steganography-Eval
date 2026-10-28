# 组员 A：服务器单图验证说明

## 先看这一段

目标是先验证“秘密信息 → PNG 图片 → 从文件提取信息”，再接组员 B 的攻击和评测模块。
本脚本不导入 `evaluation`，不计算 NIQE 或总分，也不自动下载模型。

当前状态：本地已验证消息映射、接口衔接和脚本的文件/错误处理；模型测试使用替身。
**真实 SD 生成、反演、准确率、耗时和显存占用仍待服务器实测。**
`requirements-core.txt` 是按 baseline 整理的候选组合，尚未在真实 GPU 模型环境完整验证。

下面的 Bash 命令只在租到的 Ubuntu 服务器执行。当前 Windows 电脑不用继续下载 PyTorch 或权重。

## 1. 租用前：让服务器能拿到新代码

本地开发分支为 `feature/stego`。新代码目前还未提交推送；服务器克隆远程分支会拿到旧框架。
先在本地审查 `git status` 和差异，再提交组员 A 的文件并推送此分支。
重点文件如下，不需要把虚拟环境、权重或实验图片提交到 Git：

- `core/mapping.py`、`core/steganography.py`、`core/diffusion_backend.py`
- `run_core_smoke.py`、`requirements-core.txt`
- `test_mapping.py`、`test_core.py`、`test_core_smoke.py`
- `CORE_INTERFACE.md`、`SERVER_RUNBOOK.md`、`.gitignore`
- `LOCAL_SETUP.md`、`check_gpu.py` 是已有的本地环境说明和检查工具，也可一并审查提交。

本说明不替你执行提交或推送。

## 2. 在服务器的数据盘拉取项目

以服务器的数据盘目录 `/workspace` 为例；若平台挂载路径不同，先换到实际数据盘。
以下命令均从项目根目录执行，原版 baseline 那一小节除外。

```bash
cd /workspace
git clone --branch feature/stego https://github.com/cw-zhou/Image-Steganography-Eval.git
cd Image-Steganography-Eval
git branch --show-current
git log -1 --oneline
```

核对服务器的提交与本地推送的提交一致；若看不到 `run_core_smoke.py`，先解决代码同步。

把缓存、临时文件和后续环境放到项目所在数据盘：

```bash
export STEGO_ROOT="$PWD"
export PIP_CACHE_DIR="$STEGO_ROOT/weights/cache/pip"
export TMPDIR="$STEGO_ROOT/weights/tmp"
export HF_HOME="$STEGO_ROOT/weights/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$STEGO_ROOT/weights/cache/torch"
export MPLCONFIGDIR="$STEGO_ROOT/weights/cache/matplotlib"
export CONDA_PKGS_DIRS="$STEGO_ROOT/weights/cache/conda-pkgs"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HUB_CACHE" "$TORCH_HOME" "$MPLCONFIGDIR" "$CONDA_PKGS_DIRS"
```

新开服务器终端后需要重新设置这些环境变量。项目 `weights/` 和 `outputs/` 已被 Git 忽略。

## 3. 准备 Python 3.8 和 PyTorch 2.1.0

先检查平台提供的环境：

```bash
nvidia-smi
python --version
```

如果已经提供符合要求的独立环境，可以直接使用它。否则，在已安装 Conda 的服务器中：

```bash
export STEGO_ENV="$STEGO_ROOT/weights/envs/steg38"
conda create --prefix "$STEGO_ENV" python=3.8 -y
conda activate "$STEGO_ENV"
python --version
```

若服务器提示 `conda activate` 尚未初始化，先根据平台说明初始化当前 shell；不要因此换用系统 Python。

以下选择 CUDA 12.1 构建，要求服务器驱动支持这一运行时；不要求 `nvidia-smi` 显示的版本逐字相同。
PyTorch 版本组合来自 [官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v210)。

```bash
python -m pip install numpy==1.24.4 scipy==1.10.1
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-core.txt
python -m pip check
python check_gpu.py
python -m unittest test_mapping test_core test_core_smoke -v
```

确认 GPU 检查通过、CPU 测试通过后再加载模型。此处先不安装完整的评测环境；不要用随意升级 torch 来掩盖依赖冲突。
`requirements-core.txt` 补充了原版 baseline 使用的 OpenCV；NIQE 等评测依赖另行与组员 B 对齐。

## 4. 在服务器准备本地权重

采用 [baseline 作者列出的模型来源](https://github.com/HXX5656/mas_GRDH#download-weights)：

- SD 1.5：作者提供的 [checkpoint 链接](https://drive.google.com/file/d/1ISeumzN-JrhyAacOPlh1IjAv6MKffogU/view)。
- CLIP：[openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)。

使用下面的目录约定；SD 文件重命名为示例名即可：

```text
weights/
├── sd-v1-5.ckpt
└── clip/
    ├── config.json
    ├── pytorch_model.bin
    ├── merges.txt
    ├── vocab.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── tokenizer.json
```

SD checkpoint 和整个 Diffusers 格式模型目录是不同的文件组织方式；当前加载器读取 `.ckpt` 中的权重。
不要把 SD 2.x、SDXL 或其他模型文件直接当作 SD 1.5 使用。

可在服务器用支持 Python 3.8 的 gdown 5.2.0 下载作者提供的 SD 文件：

```bash
python -m pip install gdown==5.2.0
gdown 1ISeumzN-JrhyAacOPlh1IjAv6MKffogU -O weights/sd-v1-5.ckpt
```

版本和命令参考 [gdown 5.2.0 文档](https://github.com/wkentaro/gdown/tree/v5.2.0)。
若 Google Drive 限额或服务器网络无法访问，按作者来源另行传入相同文件；不要把下载失败得到的 HTML 页面当作权重。

CLIP 只下载所需的 PyTorch 权重和 tokenizer 文件，避免把多个框架的权重都下载一遍：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir="weights/clip",
    allow_patterns=[
        "config.json", "pytorch_model.bin", "merges.txt", "vocab.json",
        "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
    ],
)
PY
```

下载目录和文件筛选方式参考 [Hugging Face Hub 文档](https://huggingface.co/docs/huggingface_hub/v0.23.5/en/guides/download)。
确认 `weights/sd-v1-5.ckpt` 和 `weights/clip/pytorch_model.bin` 是完整模型文件。

## 5. 先跑原版 baseline 的单图流程

这一步帮助区分“模型环境问题”和“我们封装的问题”。不改原始 yaml，在 `weights/` 生成当前机器专用配置：

```bash
python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
config = OmegaConf.load("baseline/configs/stable-diffusion/ldm.yaml")
config.model.params.cond_stage_config.params.version = str(Path("weights/clip").resolve())
OmegaConf.save(config, "weights/baseline_ldm.yaml")
Path("weights/smoke_prompts.txt").write_text("A photo of a cat\n", encoding="utf-8")
PY
mkdir -p outputs/original-baseline
cd baseline/scripts
python txt2img.py \
  --ckpt ../../weights/sd-v1-5.ckpt \
  --config ../../weights/baseline_ldm.yaml \
  --test_prompts ../../weights/smoke_prompts.txt \
  --dpm_gen_steps 20 --dpm_inv_steps 20 --dpm_order 2 --scale 5.0 \
  --mapping_func ours_mapping --bit_num 1 \
  --attack_layer storage --attack_factor 0.01 \
  --outdir ../../outputs/original-baseline
cd "$STEGO_ROOT"
```

`storage` 表示 PNG 保存并读取；先不加入 JPEG、噪声或中值滤波。
这里显式设置实际使用的生成和反演步数。记录终端最后的 accuracy；不要预设它一定等于 1。

## 6. 验证我们的独立封装

```bash
python run_core_smoke.py \
  --ckpt weights/sd-v1-5.ckpt \
  --clip weights/clip \
  --config baseline/configs/stable-diffusion/ldm.yaml \
  --prompt "A photo of a cat" \
  --bits "0101101000110010" \
  --output-dir outputs/core-smoke
```

封装在内存中设置 CLIP 本地路径，不需要改原始 yaml。默认运行 GPU，可用 `--device cuda:0` 指定设备。

不提供 `--bits` 时，默认生成 256 位测试消息；也可用 `--bit-length 1024 --seed 42`。
`--seed` 仅控制测试消息，不保证随机填充和整个生成过程逐像素复现。
`--bits-file message.txt` 可读取消息文件；该选项与 `--bits`、`--bit-length` 互斥。
新嵌入使用 context.version=2，支持 1～15808 位原始消息；576 位冗余长度头和填充不计入 bpp。旧 version=1 图片仍需使用原 context 解码。

每次运行会打印一个独立目录，不覆盖旧实验：

```text
outputs/core-smoke/run_时间戳_编号/
├── stego.png             # 载密图片，实际保存后重新读取
├── context.json          # prompt 和映射/采样参数，不含原始消息
├── secret_bits.txt       # 仅供实验对比的原始消息
├── recovered_bits.txt    # 模型提取结果
└── result.json           # Acc、bpp、长度、版本、耗时和可用的 GPU 显存统计
```

发生运行错误时会保存 `result.json` 和 `traceback.txt`；已成功生成的图片和 context 仍保留。
图片尚未生成时，对应文件自然不存在。参数在解析阶段就非法时，直接退出，不建立实验目录。

| 退出码 | 含义 |
|---|---|
| 0 | 有参考消息时完全一致；无参考消息时仅表示提取过程完成 |
| 1 | 提取完成，但与参考消息不完全一致 |
| 2 | 参数错误、模型加载失败或提取出错，查看错误信息和报告 |

`acc` 在长度相同时按正确位数/原始长度计算；长度不同时记 0，与当前同学评测函数的口径一致。
目标是先得到无攻击的完全恢复，但现阶段不承诺真实模型一定达到该目标。

## 7. 换一次进程，验证图片与 context 可独立提取

把刚才打印的实验目录填入 `STEGO_RUN`：

```bash
export STEGO_RUN="outputs/core-smoke/替换为刚才的run目录名"
python run_core_smoke.py \
  --mode extract \
  --ckpt weights/sd-v1-5.ckpt \
  --clip weights/clip \
  --image "$STEGO_RUN/stego.png" \
  --context "$STEGO_RUN/context.json" \
  --reference-bits-file "$STEGO_RUN/secret_bits.txt" \
  --output-dir outputs/core-extract
```

这次不会调用 `embed`。参考消息仅在提取完成后用于评分，不传给模型。
如果省略 `--reference-bits-file`，结果中的 `acc`、`exact_match`、`bpp` 为 null，表示无法核对，不能当作 100% 准确。
提取端需使用相同模型权重与配置，并使用这张图对应的 context。

## 8. 失败时先检查什么

| 现象 | 检查方向 |
|---|---|
| 缺权重、tokenizer 文件 | 本地路径、文件完整性；看 `result.json` 的 stage 和 traceback |
| 找不到模块、NumPy 转换出错 | 确认使用正确 Python 环境，运行 `pip check`；保持 NumPy 1.x 与 torch 2.1 配套 |
| cv2 提示缺少 libGL.so.1 | 在服务器补齐对应系统库，或与组员 B 统一使用相同版本的 headless OpenCV；不要同时混装两个 OpenCV 包 |
| CUDA 不可用 | `nvidia-smi`、当前解释器、GPU 版 PyTorch；服务器驱动是否支持所选构建 |
| 显存不足 | 先排除其他进程占用；当前固定单张、512×512；记录显存，不直接修改分辨率规避 |
| 权重缺失/多余参数提示 | 核对 SD 1.5 checkpoint、CLIP 和 `ldm.yaml` 是否配套 |
| 长度头无效或 Acc 偏低 | 保留 PNG、context、报告，比较原版 baseline；检查 prompt、映射种子和采样参数 |

当前长度头没有纠错保护，攻击或反演误差可能令它失效。此时应记录提取失败，不能用原文、固定比特或伪造的 Acc 补上。
在进入批量评测前，与组员 B 约定异常记录规则，避免一张失败导致整批停止。

验证后保存实际环境和代码版本：

```bash
mkdir -p outputs
python -m pip freeze > outputs/server-freeze.txt
git rev-parse HEAD > outputs/server-commit.txt
```

无攻击单图可靠后，再接三种攻击；之后跑少量 prompt，最后进行 100 条正式实验。
