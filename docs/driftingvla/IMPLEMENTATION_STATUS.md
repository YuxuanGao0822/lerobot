# Implementation status

Status of this table is static only. It does not imply successful execution.

| Component | Source location | State | Required evidence |
|---|---|---|---|
| Stock π0.5 flow baseline | `src/lerobot/policies/pi05/` | Existing implementation | Remote baseline run and eval |
| Naive π0.5 1-step | stock `PI05Policy`, inference `num_inference_steps=1` | Evaluation-only protocol | Remote latency/performance comparison |
| DBP grouping | `pi05_one_step/objectives/grouping.py` | Statically implemented with per-feature padding masks | Shape, mask and gradient tests remotely |
| DBP kernel | `pi05_one_step/objectives/drift_kernel.py` | Statically implemented with attribution and feature-mask support | Numerical/reference parity remotely |
| DBP Chunk | `PI05OneStepConfig(method="dbp_chunk")` | Statically wired | Smoke, save/reload and training remotely |
| DBP Stepwise | `PI05OneStepConfig(method="dbp_stepwise")` | Statically wired | Smoke, save/reload and training remotely |
| DriftingVLA Per-Dim | `PI05OneStepConfig(method="drifting_perdim")` | Statically wired | Smoke, save/reload and training remotely |
| Strict checkpoint load | `pi05_one_step/checkpointing.py`, policy loader | Statically implemented | `pi05_base` load/reload remotely |
| Fresh action expert | `PI05OneStepPolicy.from_pretrained` | Statically implemented with immutable `action_expert_init` provenance | Parameter-tree audit remotely |
| Optional KeyStone selector | `pi05_one_step/keystone.py`, `select_test_time_chunk` | Statically implemented; main protocol remains K=1 | K=1/4/8 selector, padding-invariance and latency tests remotely |
| RoboTwin task condition | `src/lerobot/envs/configs.py`, `robotwin.py` | Statically threaded | Easy/Hard remote environment check |
| Eval latency/resource logging | `src/lerobot/scripts/lerobot_eval.py` | Statically instrumented through task/suite/overall aggregation | Synchronized latency and peak-memory profiling remotely |
| LIBERO launcher | `experiments/driftingvla/unified_pi05/eval_libero_pair.sh` | Authored, not run | Remote paired 8-GPU eval |
| RoboTwin launcher | `experiments/driftingvla/unified_pi05/eval_robotwin.sh` | Authored, not run | Remote benchmark-server eval |
| Controlled training matrix | `experiments/driftingvla/unified_pi05/launch_train_matrix.sh` | Authored, not run | Remote dry-run review, then sequential 8-GPU execution |
| MeanFlow-VLA | external author reproduction | Intentionally not implemented | Author-supplied manifest/results |
| SnapFlow | external author reproduction | Intentionally not implemented | Author-supplied manifest/results |

All rows without remote evidence must be described as pending. No local tests or
project code have been executed.
