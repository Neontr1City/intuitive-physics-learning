# 项目结构

| 路径 | 用途 |
| --- | --- |
| `src/intuitive_physics/data.py` | 生成解析合成数据，并提供已知物理规律的参考预测 |
| `src/intuitive_physics/models/learner.py` | 公共编码器、预测器和先预测再更新的学习流程 |
| `src/intuitive_physics/models/gru.py`、`node.py`、`euler.py` | 三种状态更新器 |
| `src/intuitive_physics/models/field.py` | NODE 与 Euler 共用的向量场网络结构；训练权重各自独立 |
| `src/intuitive_physics/config.py` | 读取和检查实验配置 |
| `src/intuitive_physics/experiment.py` | 组织数据、训练模型、评估并保存结果 |
| `src/intuitive_physics/summarize.py` | 汇总已有运行的统计结果和曲线，不重新训练 |
| `configs/` | `smoke.toml` 用于试跑，`first_version.toml` 用于完整比较 |
| `tests/` | 数据、模型、实验协议与配置的测试 |
| `docs/` | 项目结构、实验设计与关键结论 |
| `.agents/skills/` | 共享的开发与整理 skills |

运行流程：**配置 → 数据生成与模型训练 → 测试评估 → 保存结果 → 跨运行汇总**。

当前训练数据按步生成，验证数据固定生成后保留在内存，测试数据保存为运行目录中的 `test_high.npz` 和 `test_low.npz`。数据是结构化数值，不是图像或仿真场景。

每次运行写入指定的 `outputs/` 子目录，包含生效配置、源码快照、模型权重（`.pt`）、测试数据、分数、`results.json` 和曲线。汇总另存 `summary.json`、`summary.txt`、`combined_results.json` 和曲线。这些输出不提交 Git；`data/` 预留给本地数据，当前尚无独立数据集。

输出按实验组织：`outputs/<实验名>/runs/` 放运行结果，`summary/` 放统一汇总，影响设计决策的预实验放 `pilot/`。同一配置的多个模型和种子尽量一次运行；分批运行也放在该实验的 `runs/` 下。纯工程试跑放 `outputs/_scratch/`，检查后清理，不另建顶层结果目录。

已有第一版结果归在 `outputs/first_version/`；其中 `archive/` 是旧工程检查和重复汇总的归档。原始配置与源码快照保留运行时内容，配置中的旧输出路径仅作历史记录。
