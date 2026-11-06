# EB-Bench

[English](README.md) | 简体中文

面向科研图像到可编辑 PowerPoint 的细粒度评测。
仓库包含固定版本的视觉复原评分器、可直接运行的完整样例，以及 208 份复原产物的实验结果。

## 快速运行

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后执行：

```sh
git clone https://github.com/BITnene465/eb-benchmark.git
cd eb-benchmark
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python"
uv sync --frozen
uv run python -B score.py examples/crossing/case.json
uv run python -B reproduce.py
uv run python -B test_core.py
```

样例已包含 PNG 渲染图和评测所需的观察记录。计算样例分数无需 API 密钥、模型服务、LibreOffice 或额外数据下载。
首次运行 `uv sync` 时，会按需下载 Python 和锁定版本的依赖。

`reproduce.py` 会重新计算四个样例，将完整评分结果与冻结版本逐项核对，并保存到 `outputs/`。

| 样例 | 源图 / 产物版本 | 预期 F 区间 |
|---|---|---:|
| `plain` | S45 / initial | 75.636666–77.579877 |
| `free` | S72 / initial | 51.347868–54.740581 |
| `shared` | S58 / initial | 61.643842–67.225909 |
| `crossing` | S82 / initial | 65.663547–66.643939 |

## 输入与输出

每份样例通过 `case.json` 指定输入，文件路径相对于该 JSON 所在目录：

```json
{
  "image": "image.png",
  "gt": "gt.json",
  "pptx": "pptx.pptx",
  "rendered": "rendered.png",
  "observations": "observations.json"
}
```

这些字段依次对应源图、源图标注、待评测 PPTX、固定渲染图，以及候选产物的观察记录。
包含自由几何特征或共享连接网络时，还需提供 `geometry` 和 `geometry_observations`。
仓库内四个样例展示了完整的数据格式。

保存单份评分结果：

```sh
uv run python -B score.py examples/plain/case.json --out outputs/plain.json
```

输出包含布局与关系 L、几何 G、外观 A、文字 T 和整图 F 的 `bounds`，以及额外内容惩罚 H、局部分数、适用项与已测项数量、输入文件哈希。
只有区间上下界相同时，`F` 才返回单个数值。尚未确定的适用项保留其权重；区间表示未解决证据带来的评分范围，不是统计置信区间。

复原模型的输入只有源图；评测侧另行使用源图标注和已核验的候选观察，包括可见对象分组、对应关系、文字转录和 rubric 判级。
评分器读取这些记录进行计算，不会仅凭任意一份 PPTX 自动生成完整标注。修改 PPTX 或渲染图后，需要重新提供与新内容绑定的观察记录。

## 代码结构

核心逻辑保留在五个平级 Python 文件中：

| 文件 | 职责 |
|---|---|
| `data.py` | 源图评测规范、候选证据、对应关系提议与 H |
| `geometry.py` | 局部对齐与容量约束边界匹配 |
| `metrics.py` | 布局、外观、文字测量与分层聚合 |
| `evaluate.py` | L/G/A/T 核心评测与原有批处理入口 |
| `render.py` | 可选的 LibreOffice 打开、渲染、保存与重开检查 |

`score.py` 是单份产物的评分入口，整合自由几何、共享网络和区间计算。
`metric.json` 固定测量参数，`lock.json` 记录评分实现的文件哈希，`reproduce.py` 用于样例复现核验。
脚本可直接运行，无需额外安装本项目为 Python 包或启动服务。

自动对应函数为 `data.automatic_correspondence`，输出用于复核的匹配提议。
在 macOS 上，默认 OCR 使用 `experiments/ocr_candidates.swift`，需要 Xcode 命令行工具。
其他平台可显式传入 `ocr_regions`；传入 `[]` 则仅使用原生对象和栅格图像提议。
已提供观察记录的数值评分不依赖这套 OCR 工具。

## 渲染新的产物

单独安装 LibreOffice，并将 `soffice` 加入 `PATH`。
macOS 上的可执行文件通常位于 `/Applications/LibreOffice.app/Contents/MacOS/soffice`。
可以直接调用现有渲染函数：

```sh
uv run python -B -c 'from PIL import Image; from render import validate_pptx; print(validate_pptx("examples/plain/pptx.pptx", "outputs/render-check", Image.open("examples/plain/image.png").size))'
```

渲染使用临时、独立的 LibreOffice 配置。字体和编辑器版本会影响像素，因此评分前需要确认新渲染图与观察记录的内容绑定一致。

## 实验结果与复现范围

`paper-results/scores.csv` 包含两类产物：

- 92 张源图上的 184 份标注辅助复原产物。
- 上述源图中 12 张图上的 24 份纯图像输入复原产物。

`reread` 列对应独立的外观观察敏感性分析。原始观察与重新观察的偏好对照表分别保留各自的比较分母。

仓库内四个样例可在 clone 后完整重算。208 行结果表是已完成实验的导出记录，不包含全部 208 份产物的原始评测输入。
重放完整实验需要研究数据归档中的源图评测规范、渲染图和候选观察。
产物有效性 V、连续编辑能力 E、修正成本 D 属于独立实验，F 评分命令不会生成这些实验成绩。

完整评分约定见 [评分协议](protocol.md)，结果表来源见 [实验结果说明](paper-results/README.md)。

## 参与贡献

Fork 本仓库，在自己的分支上修改，然后向 `main` 提交 Pull Request。
PR 中说明改动内容和验证结果。提交代码改动前，请运行：

```sh
uv run python -B test_core.py
uv run python -B reproduce.py
```

保持平级、简洁的科研代码结构。调整已冻结的评分规则前，应明确讨论，并将改动记录为新的协议版本。
