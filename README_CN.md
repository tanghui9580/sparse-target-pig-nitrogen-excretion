# 少量目标测定恢复猪氮排泄模型在独立体系中的预测能力

本仓库是论文 **“Sparse target measurements restore pig nitrogen-excretion predictions across independent settings”** 的数值复现包，包含分析数据、可执行的下游分析、模型开发阶段保留证据、机器可读结果、作图源表和锁定投稿出版层资产。

## 仓库内容

- `data/analysis_matrices/`：906条文献矩阵、546条独立生长猪目标数据，以及80条妊娠母猪评价数据（内部域/文件代码保留为 SOW80）。
- `data/design/`：包括504套平衡结果记录分配在内的确定性测定分配账本。
- `data/provenance/`：906条历史矩阵中170个 `Study_ID` 的研究级来源映射。
- `data/pre_update_predictions/`：目标 FN/UN 标签揭示前生成的预测值。
- `analysis/route/`：模型重拟合、直接迁移、目标更新、模型结构稳健性和 SOW80 评价。
- `analysis/architecture_specific/`：4种起始结构 × 7个预算 × 8种更新方法 × 504套平衡分配的模型特异性响应面选样分析。
- `analysis/classical_da_transfer_benchmark/`：补充表 S5 的经典领域自适应/迁移基准。
- `results/`：论文和补充材料所用机器可读结果。
- `figure_sources/`：主图1–5和补充图S1–S10的机器可读数值源表。
- `publication_reference/`：锁定投稿 PNG/EPS、出版源 Excel 工作簿和图题文件。
- `docs/`：变量字典、稿件—图表—结果映射和复现范围说明。

稿件与图表资产对应关系见 `docs/FIGURE_TABLE_ALIGNMENT.md`。

## 参考环境

参考环境为 **Python 3.12**；`requirements.txt` 和 `environment.yml` 均锁定依赖版本。

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.gitattributes` 统一文本文件为 LF，并把 EPS 等出版资产显式标记为 binary，避免 Windows Git 换行处理破坏文件哈希。

## 三层核验

### 1. 快速完整性/数值核验

```bash
python scripts/reproduce.py verify
```

该命令检查：

- 整包 manifest 哈希；
- 906 = 673开发 + 233确认，以及170个历史 Study_ID；
- 546 = 91种来源特异日粮 × 每种6条猪记录；
- 氮守恒关系；
- 504套平衡分配；
- P0–P3/六模型家族保留证据和选模合同；
- 生长猪与 SOW80 的关键论文数值；
- 补充表 S5 和 Figure 5 的来源合同；
- 170/170 Study_ID 来源映射与 170/170 最佳可追溯来源标识；
- 各自包含子模块中重复保存的 canonical 数据是否仍然字节一致；
- GitHub 发布卫生检查，包括缓存/pyc、100 MiB 文件门槛、依赖锁定、出版参考资产数量和发布元数据结构。

### 2. GitHub CI 使用的发布核验

```bash
python scripts/reproduce.py release-audit --jobs 1
```

在快速核验基础上继续执行：

- 336/336 模型特异性选样条目精确重建；
- light 更新方法代表性 smoke test；
- `MLR+RF + 12日粮预算 + 1个realization` 下4种 heavy 更新方法的代表性 smoke test；
- 补充表 S5 源表重建；
- 预测级重新生成并精确核对 Figure 5 的64行 RN/TNE 派生端点；
- 15幅图及锁定出版资产的一键 publication build。

`heavy-smoke` 现在是真正的快速代表性核验，不再把112个重计算单元伪装成 smoke test。完整 **112,896 cells** 仍由 `full-extension` 保留，科研分析范围没有减少。

### 3. CI 之外的完整重计算

```bash
python scripts/reproduce.py primary-factorial
python scripts/reproduce.py full-extension --jobs 8
```

预测级 `derived-endpoints` 现在已经纳入 `release-audit`。优化后的核验路径仍重新计算 504分配 × 4模型结构 × 4预算所需的选定流程更新，但不再物化庞大的逐来源预测中间文件，并跳过与 RN/TNE 核验无关的10,000次 AUBC bootstrap；科研模型结构稳健性脚本的默认完整 bootstrap 行为保持不变。

`primary-factorial` 和 `full-extension` 为完整重计算，耗时明显更高；仓库同时保留论文报告的 realization-level 矩阵，审稿人不必为核对论文数值强制重跑全部重计算。

## 一键生成论文图表

```bash
python scripts/reproduce.py publication --jobs 1
```

`generated/` 将包含：

- `rebuilt_figures/`：由机器可读源表重绘的15幅 PNG + 15幅 EPS；
- `locked_submission_artwork/`：锁定投稿的15幅 PNG + 15幅 EPS，重新物化并逐文件 SHA-256 核验；
- `tables/`：3个锁定出版源 Excel 工作簿；
- `captions/`：主图和补充图图题；
- `PUBLICATION_ARTIFACT_HASH_CHECK.csv` 与 `PUBLICATION_BUILD_REPORT.txt`。

这里明确区分“统计结果复现”和“出版排版复现”：图中的统计内容由源表重新绘制；最终字体、间距和 Excel 样式作为锁定出版层资产保存并做字节级核验，不把排版选择伪装成统计计算。

## 分析口径

文献数据库由 **673条开发记录/122个 Study_ID** 与 **233条未见研究确认记录/48个 Study_ID** 组成。模型选择完成后，以全部906条文献记录按预设模型规范重拟合，再迁移到独立目标体系。

生长猪目标体系为 **546条猪-试验期记录 = 91种来源特异日粮 × 6条记录**。3、6、9、12为预设测定预算；15、18、21仅属于扩展预算分析。

504套分配是固定数据上的84 blocks × 6 rotations，用于平衡每种日粮内6条动物记录承担已揭示 FN/UN 标签的次数，并不代表504个独立生物学重复。

P2-PAM使用测定前信息表征候选日粮空间；RSGS仅使用目标 FN/UN 揭示前生成的模型预测；CG-HPP 在更新阶段使用已揭示 FN/UN。详细变量和执行顺序见 `docs/METHODS_MAPPING.md`。

## 复现范围边界

MLR+RF 起始模型确定之后的目标域分析可由本仓库执行。P0–P3 信息比较和十模型架构选择阶段保留机器可读结果和核验材料，但本仓库**不声称**能够从原始模型开发输入端到端重新训练全部历史候选模型。详见 `docs/REPRODUCIBILITY_NOTES.md`。

## 历史文献来源追溯

`data/provenance/HISTORICAL_170_STUDY_PROVENANCE.csv` 已覆盖全部170个历史 Study_ID。来源优先级为：V44 reviewer-ready Study Index → 已核验 v35 标识 → 对前两层仍未解决的6个 Study_ID 进行出版社/期刊元数据核实。没有根据 Study_ID 字符串猜 DOI。

表中的 `REVIEW_FLAG_records` 是**记录/字段级 QA 计数**，不表示论文身份或 Study_ID 映射“不可靠”。详见 `data/provenance/README.md`。

## 许可、数据权利和引用

作者原创软件代码使用 MIT License（`LICENSE`），但该软件许可**不自动延伸**到数据矩阵、文献整理数值、投稿 artwork 或第三方来源内容；见 `DATA_RIGHTS.md`。

`CITATION.cff` 已包含发布元数据，但在真实公开仓库 URL 和永久归档 DOI 尚未生成前故意不填写这些字段。GitHub Release 和永久归档建立后，再把真实 URL/DOI 同步到 `CITATION.cff` 与稿件，不能使用虚构占位符。

正式发布顺序见 `GITHUB_RELEASE_CHECKLIST.md`；本轮公开包硬化记录见 `CHANGELOG.md`。
