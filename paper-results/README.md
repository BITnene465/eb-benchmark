# Recorded paper results

`scores.csv` has one row per artifact, with the source group, input regime,
original F bounds and the separate reread bounds when available.
It contains 208 rows from 92 distinct images in 91 source groups.

The JSON files preserve the numerical readouts used for the preference tables:

- `preference-original.json` records archived original observations and the
  separate release-coverage audit.
- `preference-reread.json` adds the blinded appearance-reread sensitivity analysis.
- `image-only.json` records the 12 image-only pairs after inventory completion.

The historical `old`, `released`, and `reread` fields identify distinct observation
or coverage states. They must not be mixed into a single denominator. Their common
48-pair intersection gives 31 concordant choices for each F state, 34 for SSIM,
and 37 for LPIPS.

Pairwise reference judgments are from two GPT visual readers. They are not human
ground truth. The 184 annotation-assisted outputs assess the evaluation method.
The separate 24 image-only outputs assess actual image-to-PPT reconstruction.

`result_pointer` in the CSV identifies the original research record, not a path
expected to exist in this compact repository. Input hashes in the example cases
are sufficient for the included four-case replay.
