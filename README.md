# Intuitive Physics Learning

认知推理课程小组项目。

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

## 共享 Skills

共享 skills 随仓库放在 `.agents/skills/`，无需复制到个人目录。在 Codex 中打开本仓库，
可自动匹配相关任务，也可以明确调用：

- `$develop-intuitive-physics`：推进研究、实验和功能开发。
- `$refactor-intuitive-physics`：保持既定行为的代码整理。

例如：`使用 $develop-intuitive-physics 实现解析数据生成器。`
如果拉取后未发现新 skills，重启 Codex 再试。外部项目记忆是个人可选配置，不随仓库共享。
发现与调用规则参见 [Codex 官方文档](https://developers.openai.com/codex/skills)。
