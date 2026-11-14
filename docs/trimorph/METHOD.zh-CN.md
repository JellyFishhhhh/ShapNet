# TriMorph-MIL：三视图分歧与合成感知患者覆盖

这是 ShapNet 上的第二项独立研究实现。入口为 `python -m trimorph`，不调用
`shapnet` 的 teacher、targets 或 student 阶段，不读取第一项工作的任何权重。
代码仅复用原项目的哈希、原子文件写入和随机种子工具。原入口保持不变。

## 输入与患者身份

每张切片提供有限、非空的 `[N,D]` 预提取特征，支持 PT 张量和 H5 的 `features`
数据集。本研究不实现 WSI 特征提取。完整视图处理全部 patch，不偷偷截断大切片。
主任务为切片二分类，患者负责划分与查询；患者的预测、三视图概率、MIL 表示和形态
组成均为其各张切片对应量的等权均值。训练以切片为单位更新，因此多切片患者提供
更多真实训练更新；这不是患者均衡损失。

必须提供带表头的清单：

```csv
slide_id,patient_id,label,feature_path
slide_a,explicit_patient_a,0,/absolute/path/slide_a.h5
slide_b,explicit_patient_b,1,/absolute/path/slide_b.pt
```

患者 ID 必须来自真实映射，不从文件名前缀推断。同患者不同标签、重复切片 ID、
复用同一特征路径、空 ID、缺失文件会报错；标签冲突必须人工解决，不做多数票。
相对特征路径相对于清单所在目录。配置中的路径相对于启动工作目录。

实验室 `label.csv` 可为无表头的 `slide_id,label`。提供独立映射后转换：

```bash
python -m trimorph.prepare \
  --features /home/users/lsy/WSI/dataset/lung_cancer_features \
  --mapping data/slide_patient_mapping.csv \
  --output data/lung_patient_manifest.csv --format h5
```

映射表必须含 `slide_id,patient_id`，与标签表完整对应；工具不会丢弃未映射的切片。
没有真实映射时，不运行真实患者划分或查询。可独立做有界健康检查：

```bash
python -m trimorph.inspect \
  --features /home/users/lsy/WSI/dataset/lung_cancer_features \
  --output runs/feature_check.json --max-slides 2 --max-patches 512
```

该命令只读取 H5 的有限片段，并通过 PT 内存映射核对对应片段，不拟合统计量、
形态字典或任务模型。它不能确认患者数、患者内标签一致性或总体数据质量。

## 标签、划分和预算协议

默认患者分层划分为训练池 60%、固定验证集 20%、测试集 20%。`split_seed` 独立于
实验种子；所有对照共享划分。也可用 `splits_file` 提供含 `train,val,test` 患者列表
的 JSON，要求全体患者恰好划分一次且每个子集具有两类。

离线实验的划分准备代码可以使用标签完成分层和冲突校验。之后 `Slide` 和
`FeatureStore` 不持有标签，只有 `Oracle` 保管隐藏标签。训练只能读取
`Oracle.revealed`；查询接口仅接受未查询的真实训练患者。验证/测试标签走单独的
评估接口。未查询标签不用于类别平衡选样、合成、原型或表示。

默认初始 15%，每轮增加 10%，最终 50%，对训练池患者数向上取整并去重。整数表示
绝对患者数；浮点数 `(0,1]` 表示比例，例如 `6` 和 `0.15`，不要用 `6.0` 表示六人。
小样本可启用 `ensure_initial_classes`：先随机揭示初始批，若已揭示标签没有覆盖两类，
再沿固定随机顺序逐个查询，最多到总查询上限。所有追加查询计入真实预算；不会扫描
隐藏标签挑选缺失类别。如果总上限内仍只有一类，按已查询的单类训练并如实记录。
追加查询可能跳过已经超过的名义预算点；导出使用实际人数，不伪造缺失预算点。
关闭该选项就是纯随机初始化。各对照同一种子使用相同初始顺序。

每轮分别记录累计训练查询人数、固定验证人数及合计。测试标签的离线评估成本不算
训练标注预算。测试预测与指标只在模型选择、形态构建、合成、查询决策完成之后计算，
不触发任何训练或超参数分支。

## 形态与网络

主协议 `queried_only` 每轮仅从当前已查询患者取样：每张切片最多 `per_slide` 个
patch，再将合并样本上限限制为 `max_fit`。标准化均值/标准差和 K-means 形态原型
均在该样本上拟合。默认 K=8；样本不足时重复中心以保持网络宽度固定。统计量和拟合
患者 ID 随轮次保存，不跨轮使用未来患者。不同轮的簇编号不要求语义一致。

显式替代协议 `train_pool_unlabeled` 允许整个训练池的无标签特征参与这些拟合；必须
另设输出目录、明确报告，不能与主协议混合。验证和测试特征从不参与任何拟合。

标准化特征经共享线性层与 ReLU 投影。与形态原型的余弦相似度经线性映射后相加。
每个视图加入可学习视图嵌入，再经共享的 `tanh × sigmoid` 门控注意力池化。
三个表示使用同一个二分类头：

- 完整视图：全部 patch。
- 判别视图：当前合法模型完整视图注意力的前 25%，加最多 32 个形态分层多样实例。
- 形态均衡视图：按形态簇轮询选取，最多 256 个实例。小簇耗尽后分配剩余额度。

子集选择为确定性操作，不对离散索引反传梯度。`importance(model,x,morphology)`
接口可替换，必须返回有限 `[N]` 分数。默认仅使用当前模型的完整视图注意力；没有
默认外部评分器，也不加载全私有标签训练的权重。

轻量线性门控对三个表示产生 softmax 权重，融合后使用同一分类头。真实切片损失为
`[CE(full,y) + fused_weight * CE(fused,y)] / (1+fused_weight)`。
判别和均衡子视图没有单独 CE，也没有预测一致性或蒸馏约束；它们通过共享参数及融合
获得梯度。融合始终包含完整视图，但其可学习权重不保证病灶保留。

可关闭形态输入、视图嵌入、门控融合或三视图。关闭融合时以完整视图读出；关闭三视图
时用完整预测重复为三份，JS 为零。`abmil` 对照还关闭形态输入与视图嵌入。网络形态
输入消融仍保留选样/合成的形态信息，以单独识别网络模块贡献。

## 信息权重与合成感知覆盖

每位患者先对各张切片同类视图的分类概率求均值，得到三份患者视图概率。令
`p_bar=(p_full+p_discriminative+p_balanced)/3`，按类别数 C 归一化：

```
H(p) = -sum_c p_c log(p_c) / log(C)
JS = H(p_bar) - [H(p_full)+H(p_discriminative)+H(p_balanced)]/3
q_j = H(p_bar_j) + alpha * JS_j
```

`entropy` 直接按 H 排序；`disagreement` 按 q 排序；`coverage` 使用 q 加权设施覆盖。
`alpha=0` 给出纯熵权重覆盖。随机选样无需对未标注池运行模型推理。

真实患者形态组成仅来自其真实 patch 簇频率，再按切片等权平均。令 MIL 表示单位化为
r，形态组成 m 开平方后采用 Hellinger 距离：

```
d_R²(j,i) = ||r_j-r_i||² / 4
d_M²(j,i) = ||sqrt(m_j)-sqrt(m_i)||² / 2
s_ji = exp[-(lambda*d_R² + (1-lambda)*d_M²)/(2*sigma²)]
c_j = max(max_{r in L} s_jr, eta * max_{g in S} s_jg)
F(B) = sum_{j in U} q_j * [max(c_j, max_{i in B} s_ji) - c_j]
```

默认 `lambda=.5,sigma=.5,eta=.25`，相似度非负，`0<=eta<1`。空参考集合的覆盖为零。
每次贪心加入边际增益最大患者，稳定 ID 顺序打破平局。批内 U、模型、表示、q、基准
覆盖均固定，包括已选入 B 的患者仍作为 U 中的被覆盖对象。没有人为排除自覆盖，
没有在选中一位后提前读取其标签或重训。完整批次选定后才一起揭示标签。

合成覆盖只是一种折扣代理参考，不获得新真实标签，不减少已查询人数，不把伪袋放进
查询池。`synthetic_coverage=false` 单独移除合成覆盖，保留合成训练。
本版精确覆盖算法具有 O(|U|²) 存储和 O(batch*|U|²) 贪心成本，适合当前数百患者规模。

## 伪袋合成与训练顺序

每轮按以下顺序运行：拟合合法形态字典 → 从固定初始化训练真实数据预热 → 用预热模型
生成训练伪袋 → 真实+合成训练并在固定验证集选最佳模型 → 用最佳模型重新生成覆盖伪袋
→ 冻结模型执行查询 → 测试评估 → 批量揭示查询标签 → 下一轮。

每轮冷启动默认使用同一份持久化初始 MIL 参数；默认是同种子的随机 MIL 初始化，输入
patch 特征已由外部提取器预训练得到。本实现没有凭空提供外部 MIL 权重。如需外部 MIL
预训练状态，指定 `external_initialization`，以严格模型 state_dict 加载，同时提供
`external_provenance` JSON：`{"source":"说明","uses_private_labels":false}`。这是一项
必须核验的使用者声明，代码无法从任意权重反推训练数据。禁止声明不实或使用看过全部
私有标签的权重。`warm_start=true` 仅为另行标识的热启动协议，继承上轮最佳参数，
重置优化器，重拟合当前形态字典。

每位已查询患者默认至多生成一袋，来源切片循环选取，供体切片随机选自同类的其他已查询
患者。没有同类供体则不生成。默认伪袋最多 1024 patch，保留其中至少 25% 的原切片最高
注意力实例，其他位置由形态分层实例填充。最多替换伪袋总量的 50%，且不得替换保护位置。
匹配模式只使用同簇供体实例；无匹配则跳过该位置，不回退成不匹配替换。普通重组模式
从同一合法供体任取实例。零替换结果不伪装成新伪袋。

每个 patch 记录来源患者、切片和原始索引，同时记录保护位置、请求/实际替换数、模式
与轮次。训练伪袋与覆盖伪袋分别保存；训练伪袋只保存一次并通过哈希绑定断点，避免每个
优化器步重复写入整库。来源标签是同类弱监督，合成不保证真实组织学合理性。合成袋使用
与真实袋相同的完整/融合损失，默认损失权重 0.5；子视图仍无独立标签约束。合成增加更新
数量，报告成本时需考虑这一点；本版不是计算量匹配实验。

## 运行、恢复与产物

建议使用已有 RRT 环境：

```bash
/home/users/lsy/miniconda3/envs/RRT/bin/python -m unittest discover -s tests -v
python -m trimorph.demo --output runs/trimorph_demo
python -m trimorph --config runs/trimorph_demo/demo.json --max-updates 2
python -m trimorph --config runs/trimorph_demo/demo.json --resume
```

主真实实验配置为 `configs/trimorph/lung.json`，默认 CUDA。小配置为
`configs/trimorph/lung_bounded.json`，默认 CPU；必须同时用 `--max-updates 2` 等边界限制
实际执行，不会把一次有限检查伪称为完整实验。完整视图在这两个配置中都保留全部 patch。

每个优化器步原子保存模型、AdamW 状态、阶段、epoch、数据游标、已揭示标签与查询审计。
实例顺序由轮次/阶段/epoch 确定，无 dropout，子视图选择与合成随机源可重建。
支持在真实预热、合成训练及轮间边界中断恢复。配置、源码、清单、全部特征及外部初始
权重的哈希检查阻止不相容恢复；特征哈希只用于完整性校验，不是模型统计量。
默认不覆盖已有非空输出目录，需显式 `--resume`。`state.pt` 是事务真源；finalize 阶段
若中断可重算相应副产物，已提交的优化器步不会重跑。同一输出目录不要同时启动两个进程。
恢复要求相同软件/设备环境；CPU 精确复现测试不等于跨硬件位级一致性保证。

保存配置、环境、固定划分、初始化、查询审计、每轮最佳模型/形态字典、训练库、覆盖库、
合成来源、选样分数、测试预测及指标。`budget_performance.csv` 同时含三种标注人数口径、
切片/患者 AUC 与 Macro-F1、合成数量、成本；固定阈值 0.5，不用测试标签校准。
`curve_area.json` 对实际查询训练患者比例积分，另给除以实际横轴跨度的面积。
单预算点没有面积，不作外推。`budget_performance.png` 可直接查看。
每轮 `seconds` 是已计时的计算段时间；`runtime.json` 记录每次活跃调用的墙钟时间，包含
哈希与断点 I/O、排除挂起间隔，并记录进程峰值 RSS；CUDA 峰值单独记录。

```bash
python -m trimorph.experiments generate \
  --config configs/trimorph/lung.json --directory runs/trimorph_suite --seeds 0 1 2
# 生成 14×3 个独立配置及顺序启动脚本，不开始训练。
# 在获得映射、准备好数据并明确安排 GPU 时间后运行：
python -m trimorph.experiments run --directory runs/trimorph_suite
python -m trimorph.experiments summarize --directory runs/trimorph_suite
```

矩阵包含随机/覆盖选择 × 无/有合成、纯熵/三视图信息值、纯熵加权覆盖、仅真实覆盖、
普通重组、形态输入/视图标识/融合/三视图消融与 ABMIL。汇总只纳入完成的运行，列出
缺失任务；各运行采用共同实际预算交集插值后计算均值、样本标准差与曲线，避免把不同
起点的面积混为一谈。缺失种子必须报告，不自动视为三种子结果。

## 大型特征库的实验 I/O

`feature_hash_cache` 默认为 null，表示每次完整计算特征 SHA256。可显式指定本地缓存
JSON；首次仍完整计算，后续仅当设备号、inode、大小、纳秒 mtime 和 ctime 均未变化时
复用该文件的 SHA256；若文件时间与首次哈希处于同一秒窗口，则保守地重新计算，
防止文件系统合并快速写入时间戳。缓存是本地可信实验材料，不应由外部修改。缓存命中不会跳过
配置、代码或特征摘要的恢复校验。修改文件即使恢复 mtime，也会因 ctime 变化而失效。
网络输出复用本次前向已计算的完整形态组成，避免在描述患者时重复编码整张切片；
数值定义、全部 patch 的使用及数据范围均保持不变。

## CUDA 与数值复现

CUDA 入口在初始化前默认设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并选择配置中指定
的 GPU。显存统计明确读取该设备，支持 `cuda:1` 等多卡编号。若默认命令环境隐藏了
NVIDIA 字符设备，需要在可访问设备的运行环境执行；仅检查到 CUDA 不可用不能判断
主机驱动故障。K-means 聚类固定单线程累加，以减少多线程中心舍入对独立重训的影响；
其余 CPU 线程数仍由 `threads` 配置。升级源码后恢复校验会要求使用相容的源码版本。
