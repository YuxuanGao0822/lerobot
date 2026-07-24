# Unified π0.5 method matrix

The main scientific comparison fixes the π0.5 architecture, pretrained checkpoint,
dataset preprocessing, action horizon, optimizer budget and evaluation protocol.

| ID | Policy type | Expert init | Teacher | Objective | Training calls | Group tensor | Inference NFE |
|---|---|---|---|---|---:|---|---:|
| `pi05_flow` | `pi05` | pretrained | no | flow matching | 1 | N/A | 10 |
| `pi05_naive_1step` | `pi05` | pretrained | no | unchanged flow; eval-only | 0 extra | N/A | 1 |
| `pi05_dbp_chunk` | `pi05_one_step` | fresh | no | DBP kernel | 1 expanded call for `G` | `[1,B,G,HD]` | 1 |
| `pi05_dbp_stepwise` | `pi05_one_step` | fresh | no | DBP kernel | 1 expanded call for `G` | `[H,B,G,D]` | 1 |
| `pi05_driftingvla` | `pi05_one_step` | fresh | no | DBP kernel | 1 expanded call for `G` | `[D,B,G,H]` | 1 |

Notation: `B` is batch size, `G` generated siblings, `H` action horizon, `D` real
action dimension. The model still uses padded width `D_max=32`; real dimensions are
sliced before the drifting kernel and invalid action timesteps are masked.
Every main row uses `test_time_samples=1`. The optional KeyStone ablation changes
only inference candidate multiplicity (`K=4` or `K=8`) and is reported separately
with candidate-equivalent compute; it is not another trained method.

MeanFlow-VLA and SnapFlow are external author-managed comparisons. Their matrix rows
are intentionally marked `TO VERIFY` rather than assigned an implementation, teacher,
or training-call count. Add them only after the author supplies a manual reproduction
manifest and remote results.

Cross-architecture π0/π0.5/SmolVLA/X-VLA/GR00T results, when available, are context
baselines and must not replace the controlled π0.5 comparison.
