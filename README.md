# 振动信号轴承故障诊断与健康趋势分析

基于 CWRU 真实振动记录的 Python 工程，覆盖数据获取、信号处理、四类故障识别、轴承特征频率、健康偏离指数、解释与扰动测试，以及 Streamlit 交互演示。

**快速配置与完整配置已实际完成 CPU 训练和评估。健康趋势模块描述记录内的健康偏离变化，不是自然退化预测或剩余寿命（RUL）预测。**

## 处理流程

```text
官方 MAT 文件 → 校验与来源清单 → 整文件/工况划分
                                ↓
                  抗混叠降采样 → 去直流 → 固定窗口
                                ↓
            统计/频谱/物理特征 → Random Forest
                      原始窗口 → 1D CNN
                                ↓
         分类评估 → 解释 → 扰动测试 → 健康指数 → Demo
```

## 数据集与下载

- 数据集：Case Western Reserve University Bearing Data Center（CWRU）。
- [官方数据说明](https://engineering.case.edu/bearingdatacenter/download-data-file)、[48 kHz 驱动端数据](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)、[正常基线](https://engineering.case.edu/bearingdatacenter/normal-baseline-data)、[轴承几何参数](https://engineering.case.edu/bearingdatacenter/bearing-information)。
- 选定 40 个 MAT 文件：4 个正常记录；内圈、外圈、滚动体故障各包含 3 种缺陷直径 × 4 种负载。外圈固定使用官方表格的 `@6:00` 位置。
- 类别：`0 normal`、`1 inner`、`2 outer`、`3 ball`。缺陷直径为 0.007、0.014、0.021 英寸，作为来源元数据保留，不输入模型。
- 信号字段：优先匹配 `X<文件编号>_DE_time`；只有一个候选字段时允许带记录的回退。RPM 只匹配已选通道前缀；缺失时使用官方标称转速并记录来源。
- 负载 0/1/2/3 HP 对应官方标称 1797/1772/1750/1730 rpm；计算优先采用实际文件 RPM。
- 故障文件属于官方 48 kHz 集合。正常文件使用 48 kHz 的明确配置约定：官方正常表格未提供逐文件采样率，MAT 也未提供可独立验证的完整采样率元数据。**不从数组长度推断采样率。**
- 全部信号经 `resample_poly` 抗混叠降采样至 12 kHz。
- 原始数据不随代码分发。官方页面可公开下载，但未找到明确的数据许可证条款；不将公开下载等同于任意再分发许可。

完整元数据见 [data/README.md](data/README.md)、[文件清单](data/manifests/cwru.csv) 和 [已记录校验和](data/manifests/checksums.json)。校验和由实际下载记录产生，不声称是数据提供方发布的签名。

```bash
python scripts/download_data.py --config configs/quick.yaml --run-dir outputs/download-local
```

下载器使用 1 MiB 分段、有限重试、续传和 MAT 解码检查，完成后原子改名。通过 `data_dir` 指定已有文件目录；缺失文件不会被合成数据替代。网络不可用时可稍后续传，或把同名原始文件放入配置目录。

源数据异常已显式处理：`99.mat` 还含 `X098` 通道，`175.mat` 还含 `X217` 通道，仅选择本文件对应字段；`174.mat` 只有 `X173` 字段且记录较短，保留该异常和官方行标签。详见 [执行修复记录](docs/execution_notes.md)。

## 安装

建议 Python 3.12，CPU 即可。GPU 可在配置中设为 `cuda`，本次未验证 GPU。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

`requirements-lock.txt` 记录本次实际使用的精确版本；需要复用版本时用它替代 `requirements.txt`。不同平台可能需要不同的 PyTorch 轮子；同版本不能保证跨硬件逐位一致。

## 快速验证

```bash
python scripts/run_all.py --config configs/quick.yaml --run-dir outputs/quick-local --download
python scripts/verify_run.py --run-dir outputs/quick-local
python scripts/check_reproducibility.py --run-dir outputs/quick-local
```

一键流程先运行自动化测试，再执行数据处理、两类模型训练、评估、解释、健康指数、鲁棒性与报告生成。每次使用新的运行目录；已存在的目录会被拒绝，防止覆盖历史结果。

## 完整实验

```bash
python scripts/run_all.py --config configs/full.yaml --run-dir outputs/full-local --download
python scripts/verify_run.py --run-dir outputs/full-local
python scripts/check_reproducibility.py --run-dir outputs/full-local
```

| 参数 | 快速验证 | 完整实验 |
|---|---|---|
| 文件集合与划分 | 相同 40 文件 | 相同 40 文件 |
| 窗口长度/重叠 | 4096 / 0% | 4096 / 0% |
| 每文件窗口数 | 最多 24 个，均匀索引选取 | 全部有效窗口 |
| CNN 最大 epoch | 8 | 50 |
| Early Stopping patience | 3 | 8 |
| RF 树数 | 100 | 300 |
| 训练种子 | 42 | 42、43、44 |

这里的完整实验覆盖预先指定的子集，不代表使用全部 CWRU 记录。两套配置均在首次模型评估前固定。

## 数据处理与泄漏防护

| 集合 | 负载 | 文件数 | 完整配置窗口数 |
|---|---|---:|---:|
| train | 0、1 HP | 20 | 415 |
| val | 2 HP | 10 | 290 |
| test | 3 HP | 10 | 290 |

快速配置为 370/240/240 个训练/验证/测试窗口；完整配置共 995 个窗口。

1. **先分文件再切窗**，同一原始记录所有窗口只属于一个集合，窗口默认不重叠。
2. 检查文件 SHA-256 与所选信号哈希，拒绝重复记录；保存选中通道、忽略通道、原始长度、RPM 来源和切窗位置。
3. 每个窗口去直流；CNN 的全局均值/标准差及特征标准化仅由训练集拟合，随模型保存。无逐窗口幅值归一化。
4. NaN/Inf、常量信号、过短信号和不合法采样率会报错；异常峰值记录质量告警，不默认裁剪冲击。
5. 默认不对原始分类输入作带通；可选 `filter_band` 必须严格位于 Nyquist 以下。包络特征采用固定 2–5 kHz 候选共振带，不称为最优频带。
6. 验证集用于最佳权重和 Early Stopping，测试集不参与参数选择。模型共享同一测试窗口清单，扰动使用固定独立种子。
7. 文件名、缺陷尺寸、负载和标签编码不作为预测特征；转速仅参与明确的物理频率计算。

**仍有边界：**跨负载文件可能来自同一轴承；记录之间存在采集差异。结果仅支持留出工况评估，不证明独立设备泛化。窗口之间存在依赖，不能把 995 个窗口当作 995 台独立设备。

## 特征与模型

17 个基础特征包括均值、标准差、RMS、峰峰值、偏度、Pearson 峭度、峰值因子、波形因子、脉冲因子、裕度因子、频谱能量、主频、频谱质心，以及 0–500、500–1500、1500–3000、3000–6000 Hz 的能量比例。分母使用数值保护；去直流前均值另存元数据。PSD 使用明确的密度归一化，并以 Parseval 一致性测试检查能量。

- **Random Forest：**基础特征 + 12 个物理特征；类别平衡权重，最小叶节点样本数 2。
- **物理消融：**保持数据、种子和树参数一致，仅去除 12 个物理特征。
- **1D CNN：**输入 `[batch, 1, 4096]`，三层带步长卷积、BatchNorm、ReLU、全局池化、Dropout 和四类分类头。
- **训练：**Adam，学习率 0.001，类别加权交叉熵；按验证集 Macro F1 选择最佳权重，平分时参考验证损失。训练损失加权、验证损失不加权，图中明确标注。
- **确定性：**固定随机种子与 PyTorch 确定性算法。RF 可并行训练，推理汇总固定单线程，避免近乎平票样本受浮点相加顺序影响。

## 轴承物理先验

令轴转频 `fr = RPM / 60`，滚动体数量 `n`，滚动体直径 `d`，节圆直径 `D`，接触角 `theta`：

```text
FTF  = fr/2 × (1 − d/D × cos(theta))
BPFO = n×fr/2 × (1 − d/D × cos(theta))
BPFI = n×fr/2 × (1 + d/D × cos(theta))
BSF  = D×fr/(2×d) × [1 − (d/D × cos(theta))²]
```

直径必须使用相同单位；配置使用官方给定的 `d=0.3126`、`D=1.537` 英寸。`n=9`、接触角为零属于与官方环故障频率倍数一致的理想化配置。标准 BSF 约为轴转频的 2.3567 倍；官方滚动体倍数 4.7135 约对应 `2×BSF`，不混淆二者。

在 Hilbert 包络谱中提取四种频率的 1/2/3 倍频附近能量比例，容差为 `max(8 Hz, 一个频率 bin)`，并在图中用不同颜色标记。转速波动、滑移、共振位置、安装路径和几何误差均可影响对齐，频率附近有能量不单独证明故障。

## 实际实验结果

完整配置，三个训练种子的均值与样本标准差；标准差不是总体置信区间：

| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- |
| 1D CNN | 80.69% ± 0.34% | 89.85% | 83.91% | 84.21% ± 0.15% |
| Random Forest + 物理特征 | 92.30% ± 2.45% | 94.68% | 93.58% | 93.52% ± 2.11% |
| Random Forest 基础特征 | 94.25% ± 2.42% | 95.49% | 95.21% | 95.21% ± 2.01% |

快速配置，种子 42；以下指标为 0–1 数值：

| model | accuracy | precision_macro | recall_macro | f1_macro |
| --- | --- | --- | --- | --- |
| Random Forest + 物理特征 | 0.8958 | 0.9313 | 0.9132 | 0.9119 |
| Random Forest 基础特征 | 0.8583 | 0.9021 | 0.8819 | 0.8809 |
| 1D CNN | 0.7875 | 0.7819 | 0.8229 | 0.7810 |

**主要观察：**完整实验中基础特征 RF 的平均 Macro F1 高于加入物理特征的版本；物理特征收益不稳定。CNN 的验证表现显著高于测试工况表现，未证明跨工况稳定性。不能把更复杂模型或更多特征等同于更好泛化。

[完整报告](outputs/full-verified/reports/experiment.md) · [逐种子指标](outputs/full-verified/metrics/comparison.csv) · [逐文件结果](outputs/full-verified/metrics/per_file.csv) · [分类报告示例](outputs/full-verified/metrics/classification_cnn_42.json) · [快速报告](outputs/quick-verified/reports/experiment.md)

![CNN 混淆矩阵](outputs/full-verified/figures/confusion_cnn.png)

![训练曲线](outputs/full-verified/figures/training_42.png)

## 可解释性与案例

- success：`217:45056`，真实类别 `inner`，预测 `inner`，置信度 0.999992。
- failure：`204:20480`，真实类别 `outer`，预测 `ball`，置信度 0.954336。

案例规则固定为种子 42 下最高置信度的正确/错误测试预测。RF 的高重要性特征包括峰峰值、频谱能量与标准差；相关特征可能分摊重要性。CNN 梯度展示当前预测 logit 对输入的局部敏感性，不是因果解释。误判案例的理论频率标记只能辅助检查，不能替模型结论背书。

![特征重要性](outputs/full-verified/figures/feature_importance.png)

![真实误判案例](outputs/full-verified/figures/case_failure.png)

## 鲁棒性测试

同一测试集，噪声 SNR 为 20/10/0 dB，连续缺失比例为 5%/10%/20%；每种设置使用 101/102/103 三个扰动种子。扰动在模型归一化之前施加，重新去直流后使用训练统计量；不重训模型。

下表为三个训练种子 × 三个扰动种子的汇总；`f1_change_mean` 为相对各自干净基线的变化，负值表示下降。

| model | perturbation | level | f1_mean | f1_std | f1_change_mean |
| --- | --- | --- | --- | --- | --- |
| 1D CNN | missing | 0.0500 | 0.8487 | 0.0057 | 0.0066 |
| 1D CNN | missing | 0.1000 | 0.8646 | 0.0102 | 0.0225 |
| 1D CNN | missing | 0.2000 | 0.8999 | 0.0094 | 0.0578 |
| 1D CNN | noise | 0.0000 | 0.7076 | 0.1010 | -0.1346 |
| 1D CNN | noise | 10.0000 | 0.8064 | 0.0248 | -0.0357 |
| 1D CNN | noise | 20.0000 | 0.8396 | 0.0063 | -0.0025 |
| Random Forest + 物理特征 | missing | 0.0500 | 0.9402 | 0.0158 | 0.0050 |
| Random Forest + 物理特征 | missing | 0.1000 | 0.9385 | 0.0104 | 0.0033 |
| Random Forest + 物理特征 | missing | 0.2000 | 0.9433 | 0.0096 | 0.0081 |
| Random Forest + 物理特征 | noise | 0.0000 | 0.7906 | 0.0103 | -0.1446 |
| Random Forest + 物理特征 | noise | 10.0000 | 0.9542 | 0.0121 | 0.0190 |
| Random Forest + 物理特征 | noise | 20.0000 | 0.9365 | 0.0193 | 0.0013 |
| Random Forest 基础特征 | missing | 0.0500 | 0.9620 | 0.0045 | 0.0099 |
| Random Forest 基础特征 | missing | 0.1000 | 0.9572 | 0.0044 | 0.0051 |
| Random Forest 基础特征 | missing | 0.2000 | 0.9368 | 0.0059 | -0.0154 |
| Random Forest 基础特征 | noise | 0.0000 | 0.8199 | 0.0063 | -0.1322 |
| Random Forest 基础特征 | noise | 10.0000 | 0.9398 | 0.0100 | -0.0123 |
| Random Forest 基础特征 | noise | 20.0000 | 0.9573 | 0.0093 | 0.0051 |

0 dB 噪声使三个模型的平均 F1 均下降。部分缺失设置反而提高了完整 CNN 的分数；这可能与移除干扰成分、幅值统计变化或该测试集的特定分布有关，尚未进行因果验证，不能推广为信号丢失有益。

![鲁棒性曲线](outputs/full-verified/figures/robustness.png)

## 健康指数与趋势

用训练集中正常窗口的 RMS、峭度、物理频带特征拟合中位数/MAD 参考。偏离指数为各特征稳健标准化后 `log1p(abs(z))` 的平均值；越高表示偏离参考越大。EWMA 仅使用当前与历史窗口，文件之间不连接。

阈值取训练正常指数的 99% 分位数。完整实验中，测试正常窗口有 **79.31%（23/29）** 超过该阈值，暴露明显工况敏感性。该阈值不适合作为已验证的实际报警规则；本项目保留这一失败结果。

没有真实退化或寿命标签，因此不生成伪造 RUL、MAE、RMSE 或 R²。时间轴是文件内的记录时间，不是设备年龄，也不按缺陷尺寸拼接生命周期。

![健康偏离变化](outputs/full-verified/figures/health_index.png)

## Demo

```bash
python -m streamlit run app/streamlit_app.py
# 或使用本地环境启动脚本
sh scripts/start_demo.sh
```

- 选择已训练实验和本地测试窗口，或输入单列数值 CSV / 一维数值 NPY；CSV 可包含 `signal` 表头。
- 提供采样率和转速。采样率不得低于工作采样率；重采样后至少需要 4096 点。
- 展示原始采集片段、处理窗口、PSD、包络谱、模型概率、特征和输入梯度。概率未经校准。
- 几何参数控件只更新理论标记，模型特征保持训练时的几何配置，避免无提示改变模型语义。
- 没有权重时显示训练命令；非法格式、NaN/Inf、过短输入和对象 NPY 会被拒绝。CSV/NPY 接口已通过实际流程检查。

```bash
python scripts/verify_demo.py
python scripts/generate_example_data.py
```

合成输入只检查接口，不进入真实训练或指标。代码包不携带原始数据与权重，首次使用需运行训练流程。

## 复核与证据

21 项自动化测试通过，覆盖物理频率、谱能量、输入检查、重采样、归一化、数据隔离、通道歧义、断点续传与模型形状。独立复核从逐条预测重新计算指标：快速版 57 份预测表，完整版 171 份。

每次运行保留配置、数据 SHA-256、源文件哈希、软件版本、逐窗口预测、训练历史、最佳权重及阶段状态。`verify_run.py` 检查数据、指标和训练参考；`check_reproducibility.py` 对全部干净/扰动条件重新推理，类别要求严格一致，浮点概率采用明确容差。

[完整复核](outputs/full-verified/reports/verification.json) · [完整重新推理检查](outputs/full-verified/reports/reproducibility.json) · [Demo 检查](outputs/qa/demo_automated.json) · [执行记录](docs/execution_notes.md)

压缩包还在新目录、新虚拟环境中重新安装并完整运行了快速流程：21 项测试通过，57 份预测表的类别一致，指标 CSV 一致。复用了相同原始文件与本机依赖缓存；这不是异机复现。[新环境复核记录](outputs/qa/clean_environment.json)

## 分阶段命令

所有阶段必须使用同一配置和运行目录，按顺序运行：

```bash
python scripts/prepare_data.py --config configs/quick.yaml --run-dir outputs/staged-local
python scripts/train_baseline.py --config configs/quick.yaml --run-dir outputs/staged-local
python scripts/train_deep_model.py --config configs/quick.yaml --run-dir outputs/staged-local
python scripts/evaluate.py --config configs/quick.yaml --run-dir outputs/staged-local
python scripts/explain.py --config configs/quick.yaml --run-dir outputs/staged-local
python scripts/health_index.py --config configs/quick.yaml --run-dir outputs/staged-local
python scripts/robustness_test.py --config configs/quick.yaml --run-dir outputs/staged-local
python scripts/generate_report.py --config configs/quick.yaml --run-dir outputs/staged-local
```

分阶段命令用于恢复中断流程；重复同一阶段会替换其阶段文件，应先保留副本或使用新的完整运行目录。

## 工程结构

```text
bearing-fault-diagnosis/
├── README.md / requirements.txt / requirements-lock.txt / pyproject.toml
├── Dockerfile / .gitignore / .streamlit/config.toml
├── configs/                 # default、quick、full
├── data/manifests/          # 官方文件清单与实际校验和
├── data/raw/                # 忽略
├── docs/                    # 协议、边界、执行记录
├── notebooks/              # 读取已完成实验的探索示例
├── scripts/                 # 下载、处理、训练、评估、复核、启动
├── src/bearing_diagnosis/   # data、signal、features、physics、models
│                            # training、evaluation、explain、health、report
├── app/streamlit_app.py
├── tests/
└── outputs/
    ├── quick-verified/      # figures、metrics、predictions、reports
    ├── full-verified/       # 同上；models、cache、logs 忽略
    └── qa/                 # 接口与复现性核对记录
```

模块采用单文件职责划分，避免空目录层级。Notebook 只读取既有结果，不承担隐藏的训练步骤。

## Docker

```bash
docker build -t bearing-diagnostics .
docker run --rm -p 127.0.0.1:8501:8501 \
  -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" bearing-diagnostics
```

本机没有 Docker 可执行程序，因此 **Dockerfile 已提供，但未完成镜像构建验证**。本机 Python、训练、测试及 Streamlit 已运行；不把它们等同于容器验证。

## 局限性与后续方向

1. 采样率与少数通道元数据需要进一步获得原始采集确认；不能以波形长度补足缺失依据。
2. 人工缺陷、同轴承跨负载和采集差异限制外推；后续使用独立设备标识进行分组评估。
3. 健康指数出现高正常误报；后续应在新的训练/验证协议内研究工况归一化，不能利用当前测试集调整阈值。
4. 物理先验并非稳定增益；候选共振带与转速误差需独立验证。
5. 若扩展真实退化预测，需加入具备连续时间和设备寿命记录的数据，再进行时间顺序划分、预测与误差评估。
6. 当前结果不建立生产可靠性、校准概率或现场安全结论。GPU、容器和独立硬件尚未验证。

## 合作成员

- [kimzclandi](https://github.com/kimzclandi)
- [Lu-Ricardo-Y](https://github.com/Lu-Ricardo-Y)
