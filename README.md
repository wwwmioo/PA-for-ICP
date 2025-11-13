# 统一多波长训练与预测包（Unified Multi-Wavelength Training & Prediction Bundle）

该文件夹包含使用“连续波长特征”（700–900nm，步长20nm）进行统一训练与预测的完整脚本与配置，支持经典监督学习模型：逻辑回归（LR）、支持向量机（SVM）、随机森林（RF）、梯度提升（GB）。

## 包含内容
- `train_unified_award.py`：统一训练脚本。固定 train/test 划分，训练四类监督模型并保存各自的模型与评估指标。
- `predict_unified_award.py`：统一预测脚本。按指定子集（train/test/custom）运行预测，生成单患者与汇总结果。
- `unified_config.json`：统一配置文件（目录、波长、聚合方式、正常范围、默认参数等）。
- `requirements.txt`：依赖清单。
- `models/`：模型保存目录（训练完成后生成 `unified_*.joblib`）。
- `results/`：输出目录（聚合表、图、指标、预测结果等）。

## 环境准备
- Python 3.8 及以上
- 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据目录结构
- `features_dir` 指向“按患者分文件夹”的特征目录，每个患者文件夹内包含各波长的 CSV，如：`700nm.csv`、`720nm.csv` … `900nm.csv`。
- 患者文件夹名包含压力信息，例如：`s2_pressure160(1)`；训练/测试的“正常/异常”标签通过 `normal_range`（默认 `[140,190]`）从压力值判定。

## 快速开始
1) 建议把输出路径改为本 bundle 目录，编辑 `unified_config.json`：
   - `output_dir`: `/storage/ruixi.sun/Data/Brain/Paper/unified_train_predict_bundle/results`
   - `models_dir`: `/storage/ruixi.sun/Data/Brain/Paper/unified_train_predict_bundle/models`

2) 训练统一模型（LR/SVM/RF/GB），使用默认连续波长：

```bash
python3 train_unified_award.py --config unified_config.json
```

3) 使用 SVM 在测试集上进行预测：

```bash
python3 predict_unified_award.py --config unified_config.json --model-name svm
``` 

4) 在训练子集或自定义患者列表上预测：

```bash
# 训练子集（需已有 split 文件，训练后生成到 results/unified_split.json）
python3 predict_unified_award.py --config unified_config.json --model-name svm --patients-file results/unified_split.json

# 自定义患者列表
python3 predict_unified_award.py --config unified_config.json --model-name svm --patients s2_pressure160(1),s23_pressure40(3)
```

## 输出说明
- 训练阶段：
  - 模型文件：`models/unified_{lr|svm|rf|gb}.joblib`，以及 `models/unified_best.joblib`
  - 聚合样本表：`results/unified_aggregates.csv`
  - PCA 图：`results/plot_unified.png`、`results/plot_unified_train.png`
  - 指标汇总：`results/unified_metrics.json`
- 预测阶段：
  - 单患者（ROI级）预测详情：`results/pred_unified_{patient}.csv` 与 `.json`
  - 汇总结果（患者级）：`results/unified_predictions_{train|test|custom}.csv`
  - 整体指标（患者级）：`results/unified_overall_metrics.json`
  - 预测 PCA 图：`results/plot_unified_predict.png`

## 重要说明
- 默认波长为连续范围 `[700, 720, …, 900]`，可在配置或命令行修改（例如只用 `760–860`）。
- 列内聚合默认使用中位数（`aggregate=median`）。
- 预测脚本支持按 ROI 范围过滤：`--roi-range 10-30`。
- 评估口径差异：
  - 训练评估：以“每患者一行”的聚合输入进行（各波长对 ROI 先做列内聚合，再拼接）。
  - 预测评估（默认）：先按 `roi_group` 跨波长对齐，逐 ROI 进行预测，再用多数投票汇总到患者级。因此训练与预测的数值可能不同。