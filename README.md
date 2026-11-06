# EB-Bench

Fine-grained evaluation of scientific figures reconstructed as editable PowerPoint slides.
This repository contains the frozen visual-fidelity scorer, complete example inputs,
and the recorded results for 208 reconstructions.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/BITnene465/eb-benchmark.git
cd eb-benchmark
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python"
uv sync --frozen
uv run python -B score.py examples/crossing/case.json
uv run python -B reproduce.py
uv run python -B test_core.py
```

The examples include their original PNG renderings and evaluator observations.
Scoring them needs no API key, model service, LibreOffice, or data download.
The initial `uv sync` downloads Python and the locked dependencies when required.

`reproduce.py` recomputes all four examples, checks the complete score payload against
the previous release, and writes the results to `outputs/`.

| Example | Source / artifact | Expected F interval |
|---|---|---:|
| `plain` | S45 / initial | 75.636666–77.579877 |
| `free` | S72 / initial | 51.347868–54.740581 |
| `shared` | S58 / initial | 61.643842–67.225909 |
| `crossing` | S82 / initial | 65.663547–66.643939 |

## Input and output

A case uses paths relative to its own `case.json`:

```json
{
  "image": "image.png",
  "gt": "gt.json",
  "pptx": "pptx.pptx",
  "rendered": "rendered.png",
  "observations": "observations.json"
}
```

Free geometric features and shared networks additionally provide `geometry` and
`geometry_observations`. The bundled cases are complete schema examples.

```sh
uv run python -B score.py examples/plain/case.json --out outputs/plain.json
```

The output contains `bounds` for L/G/A/T and F, the extra-content cost H, local
scores, eligible/measured counts, and input hashes. `F` is a scalar only when its
interval collapses. Unknown applicable requirements retain their budget.
These intervals describe unresolved evidence, not statistical confidence.

The reconstruction model receives only the source image. The evaluator separately
uses source annotations and verified candidate observations, including visible
grouping, correspondence, text and rubric grades. The supplied scorer consumes
these records. It does not obtain a complete annotation from an arbitrary PPTX.
Changed files or renderings require observations bound to the new content.

## Code

Five flat core files contain the research functions:

| File | Responsibility |
|---|---|
| `data.py` | Source contracts, candidate evidence, correspondence proposals and H |
| `geometry.py` | Local alignment and capacity-constrained boundary matching |
| `metrics.py` | Layout, appearance, text and hierarchical aggregation |
| `evaluate.py` | Core L/G/A/T evaluation and the original batch interface |
| `render.py` | Optional LibreOffice open/render/save/reopen check |

`score.py` is the portable single-submission entry, including free geometry,
shared networks and interval completion. `metric.json` fixes measurement parameters.
`lock.json` records the scoring implementation. `reproduce.py` is the example check.
There is no package installation step or service.

Automatic correspondence is available as `data.automatic_correspondence`.
Its output is a proposal for review. On macOS its default OCR uses
`experiments/ocr_candidates.swift` and requires the Xcode command-line tools.
Callers on other platforms can provide `ocr_regions` explicitly, or pass `[]`
to use only native and raster proposals. Numerical scoring does not use this OCR dependency.

## Render a new artifact

Install LibreOffice separately and put `soffice` on `PATH`. On macOS the executable
is usually `/Applications/LibreOffice.app/Contents/MacOS/soffice`.
Call the existing renderer directly:

```sh
uv run python -B -c 'from PIL import Image; from render import validate_pptx; print(validate_pptx("examples/plain/pptx.pptx", "outputs/render-check", Image.open("examples/plain/image.png").size))'
```

Rendering uses a temporary isolated LibreOffice profile. Fonts and the editor build
affect pixels, so a fresh rendering must be checked against the observation bindings
before scoring.

## Results and reproduction scope

`paper-results/scores.csv` records 184 annotation-assisted reconstructions across
92 source figures and 24 image-only reconstructions on 12 of those same figures.
The reread columns contain the separate appearance-observation sensitivity study.
The original and reread preference tables keep their own comparison denominators.

The four bundled cases can be recomputed after clone. The 208-row results table is
an exported research result, not a bundle of all 208 raw inputs. Full-cohort replay
requires the corresponding source contracts, renderings and candidate observations
from the research data archive. V, successive editing E, and repair cost D are
separate experiments and are not fabricated by the F command.

See [protocol.md](protocol.md) for the scoring contract and
[paper-results/README.md](paper-results/README.md) for table provenance.

## Contributing

Fork the repository, make changes on a branch, and open a pull request against
`main`. Describe the change and its validation. Before submitting code changes, run
`uv run python -B test_core.py` and `uv run python -B reproduce.py`.
Keep the flat research-code structure; changes to the frozen scoring rules should
be discussed explicitly and recorded as a new protocol version.
