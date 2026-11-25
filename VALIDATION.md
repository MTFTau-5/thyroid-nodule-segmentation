# 实现验证记录

验证日期：2026-11-21。环境：Tuna，PyTorch 2.14.1+cu130，SciPy 1.15.3，NumPy 2.2.6。
GPU：NVIDIA GeForce RTX 3090 Ti。所有结果仅用于实现自查，未完成正式训练。

- 真实数据：3083 图像，3070 有效 image/mask 对，13 张无 mask 排除；无相同解码灰度图。
- 固定 seed=42 的分组划分：train 2179 图/1074 患者目录，val 450/231，test 441/231。
  “患者”仅由中心和父目录标识，未核验跨中心身份；详见 SEGMENTATION.md。
- 6 项 unittest 通过：直线采样等价性、连续偏移与梯度、窗口 padding 与梯度、
  8 个模型开关组合前后向（含空前景）、原图还原与距离指标、患者与精确重复图隔离。
- 完整配置 512×512：8400857 参数；segmentation/coarse/boundary 均为 [1,1,512,512]。
- GPU FP32 推理，3 次预热、5 次测量：平均约 4.79 ms，peak allocated 约 146.8 MiB。
  单输入纯网络测量，不含图像读取/预处理/后处理，不是端到端吞吐或临床性能。
- 真实数据 GPU 混合精度训练冒烟：batch=1，2 个训练 batch、2 个验证图像，
  1 epoch；loss=2.50793；优化器、反向传播、验证与 best/last checkpoint 保存完成。
- 冒烟检查点验证 Dice=0.0，两个预测为空，HD95=null，正确报告无定义计数=2。
  这是尚未训练收敛的模型，不是有效的分割模型或实验性能。
- 同一冒烟 checkpoint 的 evaluate 与 predict 已跑通；预测二值/probability/overlay
  图像尺寸与原始图像一致。产物在 /tmp/tuna-seg-smoke，未作为训练权重发布。
- 未在实现验证时访问固定测试集进行模型选择；小批量功能评估使用 val。

已实现的六组配置参数量：baseline 7344387，global 8199683，cnn_boundary 8446403，
snake 8382805，snake_context 7545561，full 8400857。普通边界分支对照不严格等参数。

正式训练命令和评估口径见 SEGMENTATION.md。
