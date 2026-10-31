# 冻结100条COCO-caption评估

状态：本轮100条GPU评估已完成，200/200方法记录，baseline参考分74.0602、ECC候选72.1202。候选AWGN有5次长度头失败，已计零。本入口的5项本地检查通过。下文是复现与续跑说明，不要求再次运行完整评估。

## 1. 上传和计划

上传final100_evaluation_update.zip到服务器项目根目录。包里包括2个新Python文件、本文与2个冻结prompt文件，不覆盖现有算法文件。

```bash
python -m zipfile -e final100_evaluation_update.zip .
python -m unittest test_final100 -q
python run_final100.py --plan-only
```

计划为100条、baseline与payload_ecc两方法、200张生成图、800次反演；baseline16384位、候选8192位。种子从20261101开始，prompt SHA256为332419b39ada7bd659181c038df5d0a2282521da1f54f8c9670edda184f3b680。

## 2. 先运行前10条，再续跑同一计划

```bash
python -u run_final100.py --max-new-cases 10
```

首轮会流式核对7.7GB SD权重SHA256及CLIP文件哈希，需要等待；然后加载模型。只读原权重，不下载。前10条实际生成20张图、80次反演；结束应显示Completed method cases: 20/200及checkpointed。此时不公布最终总分，不能按阶段表现换样本或改算法。

确认运行没有系统错误后，在相同项目目录、相同steg38环境中续跑：

```bash
python -u run_final100.py --resume
```

默认目录固定为outputs/final100_coco_ecc。不要改名，不新建一套有利样本，不同时启动两个运行进程。若换了终端，先source weights_env.sh并激活steg38。

如需分成多个时段，可使用：

```bash
python -u run_final100.py --resume --max-new-cases 10
```

每次最多处理10个尚有待完成方法的prompt。已完成的方法不重复生成。中断用一次Ctrl+C并等程序退出；续跑时仍用--resume。突然断电也保留已原子提交的完成项；未完成项移到incomplete_*归档后按原种子重做。未完成项的重做不是替换已经完成的失败样本。

## 3. 输出和失败口径

- manifest.json：固定全部prompt、抽样来源、代码哈希、依赖、模型文件指纹和SD校验值。
- 各case目录：原图、干净及三攻击PNG、原始消息、context、提取串、初始/反演npy、错误信息；complete.json提交完成状态并记录文件哈希。
- results.json/csv：已完成方法的逐条结果。
- summary.json/status.json：全局完成数、状态和方法均值。
- 完成全部100对后才给出最终参考总分。提取失败计零；生成失败也保留，不跳过；NIQE缺失时相应方法总分为空，不伪造完整总分。
- 已完成但失败的case在续跑时也不重试。算法失败是评估结果，不应通过反复重跑挑选成功。出现系统/环境问题时保留记录再讨论处理，不自行删case。

完成后应为200/200，状态complete或complete_with_failures。后者表示存在失败/缺失指标，必须检查并报告；不是自动丢弃这些样本。已有200条完成记录时再次--resume只提示报告位置，不加载模型或重复跑图。

新入口记录同一ECC图上原始系统位与纠错后的准确率、收敛/回退块数，可分析净收益。评价采用当前B模块，JPEG70、AWGN0.05按[0,1]、中值5×5、生成图NIQE。参考消息只在解码后用于评测，不进入解码器。

模型加载或完整性检查可能在case处理前出错，保留终端报错；不能把所有系统错误都解读为算法表现。处理过程中的异常会保留已完成记录及interruption_traceback.txt。

## 4. 下载和结论

完成后先下载summary.json、results.json、manifest.json、status.json到本地“最终100条”独立文件夹，随后备份完整实验目录供报告和复查。初始/反演数值约62.5MiB，加上PNG、JSON及比特文件会更大，仍在数据盘内。

本次是项目自行选择、预先固定的新prompt评估，不是教师给定测试集，也不能保证SD训练没见过相关内容。COCO caption的来源与旧实验不同，只比较本轮同组baseline与ECC结果。若根据本轮结果再调参，本组即转为开发数据。

本地测试用替身后端，不代表已验证GPU100条运行；实际成绩以服务器保存的全部记录为准。
