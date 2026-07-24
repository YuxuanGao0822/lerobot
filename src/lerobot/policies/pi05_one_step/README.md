# Unified π0.5 one-step policies

This package holds the training-modified methods in the controlled DriftingVLA study:

- `method=dbp_chunk`
- `method=dbp_stepwise`
- `method=drifting_perdim`

Stock flow uses `policy.type=pi05 --policy.num_inference_steps=10`; naive one-step flow uses the same checkpoint and `--policy.num_inference_steps=1` at evaluation. They are deliberately not reimplemented here.

All methods inherit the stock π0.5 multimodal backbone, processor, action projections, and checkpoint key mapping. DBP Chunk, DBP Step-wise, and DriftingVLA share one generator, one drift kernel, and differ only by the grouping transform. MeanFlow-VLA and SnapFlow are intentionally not implemented in this package; they are external methods to be reproduced separately by the authors.

`test_time_samples=1` is the formal main setting. The optional KeyStone-style
`K>1` selector is inference-only and lives in `keystone.py`; it computes geometry
after slicing to the real action dimension and must be reported as additional
candidate-equivalent compute.

Status: **STATICALLY IMPLEMENTED, PENDING REMOTE EXECUTION VALIDATION.** See `docs/driftingvla/REMOTE_EXECUTION_PLAN.md` before training or evaluation.
