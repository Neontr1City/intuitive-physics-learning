# Intuitive Physics Learning

认知推理课程四人小组项目：研究物理事件经验如何改变后续预测，并比较连续与离散的学习状态更新机制。

当前状态：仓库初始化；尚未实现数据生成、模型训练或评测。

## 开发环境

使用 Python 3.11 或更新版本。在仓库根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -c "import intuitive_physics; print(intuitive_physics.__version__)"
```

目前仅提供可安装的包骨架，没有运行时依赖。数据生成和训练依赖在实现对应模块时确定并记录。

## 目录约定

- `src/intuitive_physics/`：后续数据生成、模型与评测代码。
- `pyproject.toml`：Python 包配置和依赖声明。
- 后续共享文档放入 `docs/`；实验配置放入 `configs/`；有实际验证对象后再添加 `tests/`。
- 本地数据放入 `data/`，运行产物放入 `runs/` 或 `outputs/`；这些目录不进入 Git。

提交代码与可复现实验配置；不要提交虚拟环境、大型数据、模型权重或凭据。实验实现后需记录随机种子、配置、代码提交号与结果位置。
