# Third-party notices and attribution boundary

## LeRobot / π0.5

The formal package lives in the Apache-2.0 LeRobot codebase and follows its existing
copyright headers and notices. The π0.5 implementation is the PyTorch/OpenPI-style
implementation already present in LeRobot; no new OpenPI files are copied here.

## π0.5 DriftingVLA reference

`lerobot_policy_pi05_drift-main` supplies the authoritative reference for the direct
generator, cache expansion, fresh-expert loading and grouping conventions. Its Apache
license text and copyright notice must remain discoverable in redistribution.

## DBP reference

`drift-based-policy` is used as a reference for the DBP kernel and grouping semantics.
Its MIT and inherited Diffusion Policy notices govern copied or adapted material. The
formal `pi05_one_step/objectives/drift_kernel.py` retains an attribution header and is
reorganized rather than presented as an independent original implementation.

## DBPO and external papers

The DBPO, MeanFlow-VLA and SnapFlow directories are paper sources. Their equations,
terminology and figures are cited or paraphrased; source files are not redistributed
as formal code. MeanFlow-VLA and SnapFlow are author-managed manual reproductions and
are not implemented or validated by the present package.

The optional selector in `src/lerobot/policies/pi05_one_step/keystone.py` is a
modularized Apache-2.0 adaptation of `keystone_util.py` in the authoritative
`lerobot_policy_pi05_drift-main` reference. It implements the KeyStone-style guarded
cluster-medoid selection described by Dai et al. (arXiv:2605.08638). The main
DriftingVLA protocol keeps this selector disabled with K=1.

Before release, perform a human license audit on the exact files included in the public
repository. This notice is not legal advice and does not replace license review.
