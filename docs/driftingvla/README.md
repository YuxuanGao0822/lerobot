# DriftingVLA documentation index

All material in this directory is static documentation unless explicitly labelled
`REMOTE SERVER ONLY`. No command in these documents has been executed locally.

| File | Purpose |
|---|---|
| `AUDIT_REPORT.md` | Source-grounded algorithm and repository audit |
| `IMPLEMENTATION_STATUS.md` | What is implemented, pending and intentionally external |
| `METHOD_MATRIX.md` | Controlled π0.5 methods and grouping semantics |
| `REPRODUCTION.md` | Reproduction protocol and provenance requirements |
| `REMOTE_EXECUTION_PLAN.md` | Remote-only smoke, training, evaluation and profiling checklist |
| `THIRD_PARTY_NOTICES.md` | Attribution and license boundary |

The formal package is `src/lerobot/policies/pi05_one_step/`. The legacy
`pi05_drift` package is preserved for compatibility with existing checkpoints and
experiments. MeanFlow-VLA and SnapFlow are not formal implementations here; the
authors will reproduce them separately and provide results before they are added to
the manuscript.

Static status convention:

> Statically reviewed, pending remote execution validation.
