# 医学图像分割实现

按 README 的视觉分割设计实现。当前仅处理图像与二值 mask，不读取 label1.csv，
不包含属性识别、良恶性风险、结构化事实、标签语言或报告生成。原 README 保留为研究设计。
模型输出为 segmentation/coarse/boundary 三个 **logits** 张量，不是概率；
启用多尺度上下文时，额外输出 intermediate_coarse logits 用于中间层监督。

## 环境与入口

建议使用已有的 Tuna 环境：

```bash
cd /home/mtftau-5/tuna
conda activate Tuna
python -m pip install -r requirements-seg.txt  # 已满足依赖可跳过
python -B -m unittest discover -s tests -v
```

### 代码分发包

`tuna-segmentation.zip` 解压后得到 `tuna-segmentation/`，包含分割模型、数据处理代码、
训练/评估/预测入口、配置、测试、说明文档和 `requirements-seg.txt`。
不包含原始图像、标注 CSV、患者划分清单、权重、运行产物、缓存、私有配置目录，
也不包含独立 GPU 基准脚本 `1.py` / `2.py`。这是源码包，不是离线依赖 wheel 包。

```bash
unzip tuna-segmentation.zip
cd tuna-segmentation
python -m pip install -r requirements-seg.txt
python -B -m unittest discover -s tests -v
```

迁移训练时需另外放置数据，并提供既有划分清单；只有新实验才按下文生成新划分。
预测或续训还需另外复制可信 checkpoint。GPU 环境需自行准备匹配的 PyTorch 构建和驱动。

## 数据审计与固定划分

```bash
python -m data.prepare --root data --out artifacts/splits --seed 42
```

已审计本机数据：DY 1154 对，SYFNA 930 对，YY 986 对；YY 13 张缺 mask。
共 3070 对，患者组目录划分 train/val/test=2179/450/441 张。
`artifacts/splits/audit.json` 记录排除文件、尺寸、空 mask 和中心分布；JSONL 保存相对路径。
相同解码灰度图的患者组会绑定到同一集合；相同图像若 mask 冲突则中止。
重新运行须选择新目录，防止无意覆盖固定测试集。

patient_id 取 `中心/图像父目录`，适配 DY 的额外 dydata 目录。此命名与当前数据目录吻合，
但并非经过外部患者身份库核验；跨中心同患者、近似重复图需要额外人工核查。
默认划分在每个中心内按患者组随机切分，并非外部中心验证。
外部中心验证使用：

```bash
python -m data.prepare --root data --out artifacts/holdout_DY --holdout-centre centre_DY
```

图像转换为灰度、归一化到 [0,1]，保比例 letterbox；mask 最近邻缩放。
新配置 `upscale=false`：原图小于 1024 时保持原生像素，仅补边；超过 1024 时保比例缩小。
图像缩小时使用 Lanczos 抗锯齿，mask 始终使用最近邻。这样保留更多真实细节，但不会
从低分辨率图制造新信息；1024 是否足够仍须结合病灶大小和验证集边界指标判断。
可用 `--size 768` 或 `--size 1536` 调整上限；输入尺寸须为 32 的倍数。更大画布消耗更多
显存和计算量；保留原尺寸的小图也会补到统一画布以便批处理。
翻转/小角度旋转同步用于图像与 mask。padding 不参与损失。
增强版 `configs/full.json` 还启用弹性形变、乘性散斑、增益/深度衰减与局部亮斑。
这些操作仅在训练集、letterbox 前执行，不改变验证/测试图像；弹性形变对图像与 mask
使用同一坐标场，图像双线性采样、mask 最近邻采样。强度类增强不改变 mask。
`augmentation` 中各 `*_prob` 控制概率；`elastic_alpha` 是最大位移相对图像短边的比例，
`elastic_sigma` 是平滑尺度相对短边的比例（至少 1 像素）。删除 `augmentation` 或设为
`null` 可恢复原有翻转/旋转/对比度/亮度增强；各概率设为 0 可单独禁用新增操作。
采集图存在文字、测量点叠加的可能，代码不会假设这些已清除，也不会裁掉病灶。

## 训练

```bash
python train.py --config configs/baseline.json --device cuda
python train.py --config configs/full.json --device cuda
# 小规模功能验证，不代表模型精度：
python train.py --config configs/full.json --device cuda --epochs 1 \
  --batch-size 1 --max-train-batches 2 --max-val-batches 2 --out runs/smoke
# 中断后按相同配置续训：
python train.py --config runs/full_enhanced/config.json --resume runs/full_enhanced/last.pt --device cuda
```

新配置默认画布/最长边上限 1024×1024，base=64，AdamW，Dice+BCE；粗分割权重 0.4，边界权重 0.2。
边界带从 mask 通过形态学操作生成，边界半径 2 个输入像素。无医学分类损失。
训练只访问 train/val，以验证集 Dice 保存 best.pt；训练不访问测试集。
FP16 自动混合精度仅在 CUDA 上启用，损失归约使用 FP32；梯度裁剪阈值 1。
GroupNorm 支持小 batch。GPU grid_sample 的反向传播可能非确定性；固定 seed 不是逐位复现承诺。
续训恢复模型/优化器/调度器/scaler；不承诺 persistent worker 下随机增强逐位复现。
输入分辨率、模型及训练轮数等在续训时必须与 checkpoint 配置一致。

训练/验证均有进度条，显示 batch 进度、速度、已用时间、ETA、平均 loss、segmentation/
coarse/boundary 分项损失、Dice 和 IoU；训练另显示学习率、裁剪前梯度范数、GPU 峰值显存。
训练 Dice/IoU 在增强后的输入网格计算，屏蔽 padding；验证指标在原图尺寸计算，两者口径不同。
终端覆盖刷新；重定向日志默认每 10 个 batch、首批和末批输出一行。
`--log-every 1` 使非终端日志逐批输出；`--no-progress` 关闭进度条，保留 epoch 汇总。
进度条使用标准库实现，无需安装 tqdm。

输出：config.json、environment.json、history.jsonl、last.pt、best.pt。
history.jsonl 新增 train_losses、train_dice、train_iou、lr、val.losses 字段。
混合精度保持原设置：CUDA 训练为 FP16 autocast + FP32 损失及 GradScaler，验证仍为 FP32，
未切换到 BF16。旧 checkpoint 缺少新预处理字段时按旧的放大/双线性策略解释，预测和评估
沿用该 checkpoint 的配置；续训请使用运行目录保存的 config.json，不能直接混用新配置。
新的高分辨率训练建议指定新目录，例如：

```bash
python train.py --config configs/full.json --device cuda --out runs/full_1024
```

已运行的 Python 训练进程不会自动加载代码变更，已有权重/日志也不会在本次修改中更改。
使用 `--max-*-batches` 的运行会在 environment.json 标记 smoke_run；不得视为正式指标。
数据、模型权重和实验产物不提交到代码仓库。

## 网络和消融

当前增强版的整体结构图见 `docs/model_architecture.png`，细化模块展开图见
`docs/boundary_refinement.png`；同目录提供 SVG 矢量图、DOT 绘图源文件和
`docs/model_architecture.md` 尺寸/连接说明。

- CNN encoder: 输入 /4、/8、/16、/32，通道 64/128/256/512。
- 全局 Transformer: 最深层投影到 256 通道，加可插值位置嵌入。
- U-Net decoder: 三层 skip，输出 /4 特征。
- DSConv: x/y 初始方向的连续累计偏移采样；中心固定，所有非中心偏移保持可微。
- context: 预测粗分割构造内部/边界/外围软角色；默认 detach 角色图。
- 局部多头交叉注意力: **不重叠 9×9 窗口**、相对位置偏置、padding key 屏蔽。
  是 README 局部邻域的一种实现选择，不等同于每个像素中心的滑动 9×9 邻域；窗口边界
  不能直接互相读取，远距离关系由深层全局注意力承担。
- 门控投影与残差融合；精细 mask 与边界输出回到输入尺寸。

### 增强版 full 配置

- `depth=3`：瓶颈全局注意力使用三层，可改为 2 做消融。实际 encoder 下采样率为
  **32**，因此 1024 输入的瓶颈是 **32×32、1024 tokens**，不是 64×64。
- `multiscale_context=true`：在 decoder 的 /8 层增加一次上下文细化模块；1024 输入时
  是 **128×128**，末端 /4 层仍是 256×256。两个尺度有各自的蛇形分支与窗口交叉注意力。
  中间粗分割头接受显式监督，避免 `detach_roles=true` 时没有训练信号；两个 coarse
  Dice+BCE 损失取平均后乘原有 `coarse_weight`，不额外翻倍粗分割损失权重。
- `refinement_steps=3`：每个尺度共享其自身细化模块权重迭代三次，将上一步残差细化
  特征作为下一步输入，重新预测蛇形偏移；同一尺度的 coarse 角色图保持不变。
  参数量不会随步数倍增，但计算量和训练激活显存会增加。该设置仅影响 snake/context。
- 本次选择迭代细化方案，**没有新增拓扑约束损失，也不保证采样链不自交或拓扑正确**，
  不能将其描述为完整 DSCNet 或其拓扑损失的复现。

新增模型参数的构造默认值仍为 `multiscale_context=false`、`refinement_steps=1`，
`depth` 默认仍为 1，以兼容旧 checkpoint；只有增强版 full 配置显式启用新结构。
其它消融配置保持不变。新增结构需要重新训练，不能用旧 full 权重直接严格加载新结构；
旧模型预测/续训应继续使用 checkpoint 自带或原运行目录保存的配置。
full 的默认输出改为 `runs/full_enhanced`，避免与旧 `runs/full` 混用。
比较消融时请统一增强策略，并分别控制 depth、多尺度与迭代步数，不能将新旧 full
之间的差异归因于单一模块。显存不足时先用 `--batch-size 1` 或 `--batch-size 2`。

配置对应：baseline（纯 CNN）、global、cnn_boundary（普通分支容量对照）、snake、
snake_context（无全局注意力）、full。cnn_boundary **不是严格等参数量**，比较时须记录
参数量、训练预算和实际耗时；一般可变形卷积、上下文指导偏移属于未实现的后续实验。
DSConv 是连续偏移思想的独立实现，不是对 DSCNet 官方权重/输出逐位兼容的复刻。
不使用管状拓扑损失或强轮廓平滑。

## 评估与预测

```bash
python evaluate.py --checkpoint runs/full_enhanced/best.pt --split val --device cuda \
  --out artifacts/val
# 阈值只能在验证集选择，固定后再运行测试：
python evaluate.py --checkpoint runs/full_enhanced/best.pt --split test --device cuda \
  --threshold 0.5 --tolerance 2 --out artifacts/test --save-masks
python predict.py --checkpoint runs/full_enhanced/best.pt --input data/centre_YY/images \
  --device cuda --out artifacts/predictions
python profile_model.py --device cuda --size 512 --base 64
```

评估和预测均可显式开启 TTA，训练期验证默认不使用 TTA：

```bash
python evaluate.py --checkpoint runs/full_enhanced/best.pt --split val --device cuda \
  --tta-flip --tta-scales 0.75 1 1.25 --out artifacts/val_tta
python predict.py --checkpoint runs/full_enhanced/best.pt --input data/centre_YY/images \
  --device cuda --tta-flip --tta-scales 0.75 1 1.25 --out artifacts/predictions_tta
```

TTA 将水平翻转预测翻回，再将各尺度的 sigmoid **概率**缩放回基础输入网格取平均，
最后去除 letterbox 并还原原图尺寸；不是平均阈值化 mask 或 logits。
除原尺度保留原尺寸外，缩放尺寸取最近的 32 倍数，最小 32。参数缺省为原尺度单次推理，
`--tta-flip` 单独使用为两次推理；三尺度加翻转为六次，绝非零耗时成本。
所用 TTA 参数记录在评估指标和预测元数据中；评估耗时包含全部 TTA 前向。
启用 TTA 的验证损失（若通过 Python 接口传入 criterion）仍来自原图单次前向。
是否提升 Dice 必须通过验证集实验确认，先固定 TTA 和阈值，再评估测试集。

evaluate 在原图分辨率算 Dice、IoU、HD95、Boundary F1 和 Surface Dice；距离单位是
**原图像素**，容差默认 2 px。缺少可靠像素间距时不能换算 mm。若预测或真值仅一方为空，
Dice/IoU/边界分数为 0，HD95 为 null，并报告无定义样本数；不会静默记为 0。
提供逐图、逐中心汇总和按患者等权 bootstrap 的 Dice 95% 区间；总体 mean 是逐图平均，
与患者等权均值不是同一口径。小结节等困难病例分层需预先定义规则后另行实验。
`--limit` 只用于冒烟，输出 partial_evaluation=true。

predict 输出原尺寸 .mask.png（二值）、.prob.png（uint16/65535 为概率）、.overlay.png
和几何变换元数据。不输出分类标签或医学描述。仅加载可信的本项目 checkpoint。

## 实现验证与限制

已用本机真实数据完成审计与小批量训练/评估/预测功能验证（详见 VALIDATION.md）。
这些检查证明代码链路可执行，不代表收敛、研究假设成立或具备临床精度。
正式实验需按患者固定测试集、多 seed、消融比较并记录边界困难样本表现。
未自动运行耗时完整训练，未生成或声称已获得训练好的有效模型。

参考：连续蛇形采样来源于 [DSCNet, ICCV 2023](https://github.com/YaoleiQi/DSCNet)。
