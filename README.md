# Intuitive Physics Learning

认知推理课程小组项目，使用 Python 3.12 和 uv。

## 快速开始

先[安装 uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在仓库根目录运行：

```bash
uv sync --locked
uv run python -c "import intuitive_physics; print(intuitive_physics.__version__)"
```

uv 会准备 Python 和 `.venv` 环境，不需要手动激活。目前只有项目骨架，实验代码后续加入。

## 日常使用

- 拉取代码后：`uv sync --locked`
- 运行脚本：`uv run python 脚本路径.py`
- 添加依赖：`uv add 包名`，并一起提交 `pyproject.toml` 和 `uv.lock`。

代码放在 `src/intuitive_physics/`；本地数据放在 `data/`，实验输出放在 `outputs/`。环境、数据和实验输出不提交 Git。
