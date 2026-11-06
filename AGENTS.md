# Research-code boundary

- Keep the five flat core files: data.py, geometry.py, metrics.py, evaluate.py, render.py.
- score.py is the portable F entry; reproduce.py checks the bundled cases. Avoid services, frameworks, compatibility layers, and extra infrastructure.
- Manage Python and dependencies with uv in .venv and .runtime. Do not alter the global environment.
- metric.json and lock.json identify the frozen scoring rules. A change to the scoring protocol requires explicit agreement and a new recorded version.
- Never add sample IDs, model names, or hand-picked output exceptions to metric logic.
- Keep original images, PPTX files, and annotations unchanged. Changed inputs require new hash-bound observations.
- Missing predictions and unresolved evidence are different states. Do not discard unresolved source budgets.
- Verified observations are inputs to the scorer; automatic correspondence is a proposal, not complete ground truth. Model-reader judgments are not human judgments.
- Validate refactors with all four complete score payloads and the core checks. V, E, and D remain separate from the F command.
