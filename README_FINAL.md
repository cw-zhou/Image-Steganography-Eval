# 最终方案运行说明

最终候选：公开头部随机化与稀疏消息纠错，保留baseline作参照。核心算法及实验入口17个源文件hash与完成的100条评估manifest一致，见上级目录代码核对清单.json。文件包不包含权重、环境或缓存。

## 已完成结果

100条COCO描述、200张生成图、800次提取。baseline课程公式参考分74.0602，候选72.1202。候选有效bpp为0.03125，baseline为0.0625。候选无攻击/JPEG/AWGN/中值Acc分别99.8147%、94.2269%、82.1050%、91.9225%。AWGN有5次长度头失败，均计零。不要引用开发10条分数作为最终结果。

## 环境和权重

Python3.8.20、torch2.1.0+cu118、torchvision0.16.0+cu118；其余版本见requirements-zhejiang.txt及zhejiang_environment_freeze.txt。实际环境版本也保存在最终实验manifest。不要升级torch来掩盖版本冲突。

模型路径约定：weights/v1-5-pruned.ckpt、weights/clip/、weights/cache/torch/hub/checkpoints/niqe_modelparameters.mat。SD SHA256为e1441589a6f3c5a53f5f54d0975a18a7feb7cdf0b0dee276dfc3331ae376a053。模型和缓存放数据盘；本地使用D盘。当前四川服务器已经配好，无需重装。

## 测试

```bash
python -m pip check
python check_gpu.py
python -m unittest test_sparse_payload_code test_payload_ecc_candidate test_payload_ecc_batch2 test_mapping test_core test_core_smoke test_final100 -q
```

本地替身与纯数值测试共53项通过；GPU证据为最终100条记录，不能混称为53项GPU测试。

## API

```python
import json
from pathlib import Path
from PIL import Image
from ecc_payload_candidate import ECCPayloadCore
from run_robustness_pilot import seed_everything

core = ECCPayloadCore(
    'weights/v1-5-pruned.ckpt', 'weights/clip',
    'baseline/configs/stable-diffusion/ldm.yaml', device='cuda')
seed_everything(123)
image = core.embed('01011000', 'A photo of a cat', padding_seed=123)
image.save('example_stego.png')
Path('example_context.json').write_text(
    json.dumps(core.get_context()), encoding='utf-8')

# 可在新进程重新创建core，仅提供图像和context进行提取。
context = json.loads(Path('example_context.json').read_text(encoding='utf-8'))
seed_everything(323)
with Image.open('example_stego.png') as stored:
    recovered = core.extract(stored.convert('RGB'), context=context)
print(recovered)
```

此例仅说明API，需要GPU和本地权重，不承诺任意图像/消息均完全恢复。长度1—8192位，超长拒绝。提取器没有reference参数，context不含秘密或实际消息长度。

## 最终评估入口

```bash
python run_final100.py --plan-only
```

已有完整运行时不要重跑。原目录outputs/final100_coco_ecc可用以下命令核验完成状态，完成200条后不会重复生成：

```bash
python -u run_final100.py --resume
```

若仅为在新环境独立复现，保留固定prompt与代码，以新目录运行并明确标记为复现，不能挑选较高分作为原始最终结果：

```bash
python -u run_final100.py --run-dir outputs/reproduction_final100
```

共生成200张图、反演800次，会产生GPU费用。没有必要为交报告重复执行。

最终实验100条在生成前固定；根据该结果再修改算法会使其转为开发数据。原作者参考使用统一封装后端，非逐字节重放txt2img CLI；容量不同，需报告其取舍。

## 来源

- 原作者代码 https://github.com/HXX5656/mas_GRDH
- 项目共享仓库 https://github.com/cw-zhou/Image-Steganography-Eval
- COCO描述来源 https://github.com/salesforce/BLIP/blob/main/data/coco_karpathy_dataset.py
- 归一化min-sum参考 https://aff3ct.readthedocs.io/en/latest/user/simulation/parameters/codec/ldpc/decoder.html

旧V3及其他实验脚本仅用于过程记录；当前候选API为ECCPayloadCore，run_core_smoke.py默认的旧core不能自动当作最终ECC算法使用。
