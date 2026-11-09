# Intuitive Physics Learning

认知推理课程小组项目，研究经验如何改变物理预期，以及不同状态更新方式对学习与迁移的影响。目前已完成一版小型解析合成实验。

## 快速上手

先[安装 uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在仓库根目录运行：

```bash
uv sync --locked
uv run python -m intuitive_physics.experiment --config configs/smoke.toml --output outputs/_scratch/smoke
```

uv 会准备 Python 3.12 和虚拟环境，无需手动激活。当前实验使用 CPU，无需物理引擎。
以上命令仅做两步试跑；输出目录必须是新目录，再次运行时请更换名称。

## 文档

- [项目结构](docs/structure.md)：各模块、配置和输出的用途。
- [第一版实验](docs/first-experiment.md)：基本设计、运行方式、关键结果与当前结论。

## 协作开发

拉取代码后运行 `uv sync --locked`。添加依赖使用 `uv add 包名`，一起提交 `pyproject.toml` 和 `uv.lock`。
本地数据、虚拟环境和实验输出不提交 Git。

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

仓库在 `.agents/skills/` 提供两套共享 skills，可在 Codex 中调用：

- `$develop-intuitive-physics`：推进研究、实验和功能开发。
- `$refactor-intuitive-physics`：保持既定行为的代码整理。
