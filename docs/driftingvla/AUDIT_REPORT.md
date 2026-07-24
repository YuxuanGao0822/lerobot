# DriftingVLA Static Audit Report

Status: **STATICALLY AUDITED — REMOTE EXECUTION REQUIRED.** No project module,
checkpoint, simulator, benchmark, test, or LaTeX compiler was executed locally.

## 1. Evidence map and license boundary

| Directory | Role | Evidence inspected | License/attribution note |
|---|---|---|---|
| `lerobot` | Formal implementation target | `src/lerobot/policies/pi05/*`, policy registration, dataset/env and eval scripts | Apache-2.0; retain existing notices |
| `lerobot_policy_pi05_drift-main` | π0.5 DriftingVLA reference | six files under `src/lerobot_policy_pi05_drift/` | Apache-2.0 text; preserve copyright/attribution |
| `drift-based-policy` | DBP reference | `drifting_unet_lowdim_policy.py`, `drifting_util.py`, workspace | MIT/Diffusion Policy notices; do not copy unrelated code |
| `Drift_Based_Policy_Optimization_*` | DBPO paper source | `secs/3_preliminaries.tex`, `secs/4_method.tex` | source-only citation/reference |
| `arXiv-2603.01469v1` | MeanFlow-VLA source | `root.tex`, `Mybib.bib` | external manual reproduction only |
| `arXiv-2604.05656v1` | SnapFlow source | `latex_arxiv.tex` | external manual reproduction only |
| `TRLtemplate` | T-RL manuscript target | IEEE template files | preserve original template; Overleaf only |

The six Python files in the existing `lerobot/src/lerobot/policies/pi05_drift`
package are byte-identical to the supplied standalone reference at audit time.
That compatibility path is not modified by the unified package.

## 2. Stock π0.5 flow path

Evidence: `src/lerobot/policies/pi05/modeling_pi05.py` and
`src/lerobot/policies/pi05/configuration_pi05.py`.

- `PI05Pytorch.__init__` builds a PaliGemma VLM and Gemma action expert with
  action input/output projections and AdaRMS time conditioning.
- `embed_prefix` encodes image and language tokens. The proprioceptive state is
  serialized into the language prompt by `PI05PrepareStateTokenizerProcessorStep`
  in `processor_pi05.py`.
- `embed_suffix` consumes a padded action/noise chunk `[B,H,32]` and a scalar
  time embedding.
- `forward` uses `x_t=t\epsilon+(1-t)A` and velocity target `\epsilon-A`.
- `sample_actions` builds a prefix KV cache once and calls `euler_integrate`; the
  default `num_inference_steps` is 10.
- `PI05Policy` owns preprocessing, action normalization, padding/slicing,
  execution queue and checkpoint conversion.

Expected core shapes are prefix `[B,L_p,W]`, action `[B,H=50,D_max=32]`,
velocity `[B,H,32]`, and returned action `[B,H,D_real]`. `action_is_pad` is
`[B,H]`; stock π0.5 does not apply the explicit padded-timestep mask used by the
DriftingVLA loss, so this difference must be reported in controlled comparisons.

## 3. Authoritative DriftingVLA behavior

Evidence: `lerobot_policy_pi05_drift-main/src/lerobot_policy_pi05_drift/modeling_pi05_drift.py`
and the identical in-tree compatibility copy.

- `_build_prefix_cache` encodes the multimodal prefix once and retains gradients.
- `_expand_cache` uses observation-major `repeat_interleave` to create `B*G` cache entries.
- `_sample_one_step` evaluates noise at fixed `t=1` and directly reads the output as
  an action chunk; it does not subtract a velocity or run Euler integration.
- `_sample_one_step_n` returns `[B,G,H,D_max]` from one expanded expert call.
- `forward_drifting` slices real action coordinates before grouping and calls the
  shared grouped drifting kernel.
- Existing groupings are Chunk `[1,B,G,H*D]` and Step-wise `[H,B,G,D]`; the proposed
  VLA extension is Per-Dimension `[D,B,G,H]`.
- Invalid action timesteps are masked before grouping. The same direct generator is
  used for training and inference.
- `fresh_action_expert` excludes the Gemma expert, action projections and time MLP
  from the pretrained state load, while retaining the VLM. It serializes provenance
  through `init_label`.
- Optional KeyStone test-time selection from the authoritative reference is migrated
  to `pi05_one_step/keystone.py` and `select_test_time_chunk`. The controlled core
  setting uses one direct sample (`K=1`); `K>1` is an explicitly labeled ablation.
  Selection distances use only real action coordinates, while sibling candidates
  share one prefix cache and one batched action-expert invocation.

## 4. Original DBP/DBPO kernel

Paper evidence: `secs/3_preliminaries.tex` and `secs/4_method.tex` in the DBPO
source. Code evidence: `drift-based-policy/diffusion_policy/model/drifting/drifting_util.py`
and `drifting_unet_lowdim_policy.py`.

The kernel uses generated siblings as negatives and a demonstration as a positive.
It computes pairwise L2 distances, data-dependent scale, scale-aware normalization,
self masking, multi-temperature symmetric affinities, balanced attraction/repulsion,
force RMS normalization, and a stop-gradient target. Chunk flattening and Step-wise
temporal slicing are already prior work. DBPO's separate online PPO likelihood adapter
is not silently claimed by the current π0.5 offline post-training package.

The formal unified implementation is:

```text
src/lerobot/policies/pi05_one_step/
├── configuration_pi05_one_step.py
├── modeling_pi05_one_step.py
├── processor_pi05_one_step.py
├── checkpointing.py
├── keystone.py
└── objectives/
    ├── drift_kernel.py
    └── grouping.py
```

`drift_kernel.py` is a reorganized Apache-attributed implementation of the DBP
kernel. `grouping.py` is the only method-specific transformation: the network,
prefix cache, sibling generation and kernel remain shared. The formal path adds a
per-feature validity mask so padded timesteps are removed from distances, scale,
force RMS, and residual normalization instead of being treated as zero-valued
features. This padding correction is inactive when every timestep is valid and must
be checked against physically truncated tensors on the remote server.

## 5. Controlled method matrix

| Method | Expert init | Teacher | Objective | Training calls | Grouping | Inference |
|---|---|---|---|---:|---|---|
| π0.5 flow | pretrained | no | flow matching | 1 | full-chunk velocity | 10-step Euler |
| π0.5 naive 1-step | pretrained | no | unchanged flow | 0 extra | N/A | 1 Euler update |
| DBP Chunk | fresh | no | shared DBP kernel | 1 expanded call for `G` | `[1,B,G,HD]` | 1 direct read |
| DBP Stepwise | fresh | no | shared DBP kernel | 1 expanded call for `G` | `[H,B,G,D]` | 1 direct read |
| DriftingVLA | fresh | no | shared DBP kernel | 1 expanded call for `G` | `[D,B,G,H]` | 1 direct read |
| MeanFlow-VLA | TO VERIFY | TO VERIFY | external manual reproduction | TO VERIFY | TO VERIFY | 1 |
| SnapFlow | TO VERIFY | TO VERIFY | external manual reproduction | TO VERIFY | TO VERIFY | 1 |

MeanFlow-VLA and SnapFlow are intentionally **not implemented** in this formal
package. The supplied LaTeX sources remain useful for the author’s independent
reproduction and for accurate Related Work/comparison entries. No associated result
may be inserted into the manuscript until the author supplies the implementation
manifest and remote results.

## 6. Checkpoint and initialization contract

`PI05OneStepPolicy.from_pretrained` requires an explicit `PI05OneStepConfig`,
loads `model.safetensors`, applies π0.5 key normalization, refuses silent fallback
to random weights, and checks missing/unexpected keys. For `fresh_action_expert=True`,
the VLM is warm-started while the action expert/projections/time MLP are reset.
The one-time switch is cleared after initialization, while the immutable
`action_expert_init` and `init_label` fields retain provenance so a trained checkpoint
does not reset its expert when reloaded. Actual checkpoint loading, save/reload, and
optimizer compatibility remain remote requirements.

## 7. Central novelty boundary

DBPO already contains distribution drifting, native one-step generation, sibling
attraction/repulsion, Chunk mode and Step-wise mode. The journal extension must not
relabel those as first contributions. The new claim is narrower and testable:

> Under a fixed pretrained π0.5 VLA, treating each real action coordinate as an
> independent temporal drifting unit can improve the accuracy/efficiency trade-off
> of native one-step closed-loop control relative to chunk-level and stepwise grouping.

The transformer representation remains shared. The new factorization is applied to
the drifting loss geometry, not asserted to be an independent action policy.

## 8. Static uncertainties and required remote evidence

1. Confirm exact gradients through prefix-cache prefill and expanded cache in full
   fine-tuning mode.
2. Confirm that the three groupings produce exactly the documented shapes and that
   padded timesteps have zero contribution and zero generated-sample gradient.
3. Confirm state-dict filtering, fresh-expert initialization, checkpoint reload and
   processor compatibility from `pi05_base`.
4. Confirm actual forward-call counts and latency with device synchronization.
5. Confirm DDP, bf16, gradient checkpointing, optimizer memory and 8-GPU behavior.
6. Confirm LIBERO and RoboTwin 2.0 benchmark outputs, seeds, checkpoint selection and
   real action dimensions.
7. MeanFlow/SnapFlow reproduction and all associated numbers are external manual
   work, not evidence from this package.

Every item above is **pending remote execution validation**. No statement in this
report means that a model has trained, evaluated or compiled successfully.
