# Third-party notices

The project license in [`license.txt`](license.txt) does not replace the licenses
of incorporated third-party material. The following notices cover distributed
source, not all installed packages, datasets, or downloaded model weights.

## MultiObjectiveOptimization — MIT

- Scope: `src/criterions/loss_weighting/min_norm_solvers.py` and the MGDA method
  in `src/criterions/loss_weighting/weight_methods.py`.
- Copyright (c) 2018 Ozan Sener.
- [Full license](LICENSES/MultiObjectiveOptimization-MIT.txt).
- [Upstream license](https://github.com/isl-org/MultiObjectiveOptimization/blob/bf5bbb30371489a47f6440691d8934b70f7dfc3b/LICENSE).
- Research: Ozan Sener and Vladlen Koltun, *Multi-Task Learning as Multi-Objective
  Optimization*, NeurIPS 2018. The local solver also incorporates the Nash-MTL
  downstream version; both notices are retained.

## Nash-MTL — MIT

- Scope: loss-weighting framework and methods in
  `src/criterions/loss_weighting/weight_methods.py` and the downstream solver in
  `src/criterions/loss_weighting/min_norm_solvers.py`.
- Copyright (c) 2022 Aviv Navon.
- [Full license](LICENSES/Nash-MTL-MIT.txt).
- [Upstream license](https://github.com/AvivNavon/nash-mtl/blob/cce18403ef5557a6b30f4ba43b896117107d6902/LICENSE).
- Local adaptations include relative imports, the factory interface, task-count
  checking, and gradient-retention behavior. Individually attributed methods
  retain the notices below.

## Pytorch-PCGrad — BSD 3-Clause

- Scope: PCGrad in `src/criterions/loss_weighting/weight_methods.py`.
- Copyright (c) 2021, Wei-Cheng Tseng. All rights reserved.
- [Full license](LICENSES/Pytorch-PCGrad-BSD-3-Clause.txt).
- [Upstream license](https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/e987ac603fa1accd386820a985a6dc2fd92dec5b/LICENSE).

## MTAN — MIT

- Scope: DynamicWeightAverage in
  `src/criterions/loss_weighting/weight_methods.py`.
- Copyright (c) 2020 Shikun Liu.
- [Full license](LICENSES/MTAN-MIT.txt).
- [Upstream license](https://github.com/lorenmt/mtan/blob/c36c30baa18968dec74fe9039abcfd4f132edfa1/LICENSE).

## Multi-task learning example — MIT

- Scope: Uncertainty in `src/criterions/loss_weighting/weight_methods.py`.
- The upstream notice is literally `Copyright (c) 2017 `, without a named holder;
  the bundled license preserves the original text rather than inventing one.
- [Full license](LICENSES/Multi-Task-Learning-Example-MIT.txt).
- [Upstream license](https://github.com/yaringal/multi-task-learning-example/blob/86e2c882a04aa217b76796d67a0026fb3db4ff24/LICENSE).

## Segmentation Models PyTorch — MIT

- Scope: adapted FPN construction, heads, preprocessing, and decoder code in
  `src/models/FPN.py`.
- Copyright (c) 2019, Pavel Yakubovskiy.
- [Full license](LICENSES/Segmentation-Models-PyTorch-MIT.txt).
- [Upstream historical license](https://github.com/qubvel-org/segmentation_models.pytorch/blob/a288d337821716ad67125127b5dd96a1cd833391/LICENSE).
- Local changes include multitask parameter access, grouped/custom heads,
  decoder checkpointing, and tensor/per-channel preprocessing.

## Existing FPN attribution — CC BY-NC 4.0

`src/models/FPN.py` retains its attribution to Davy Neven and its
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) notice.
The full license is in [`license.txt`](license.txt). This attribution does not
replace the SMP notice for incorporated SMP material.

Pinned upstream revisions identify the retrieved license texts; they do not
claim that the original adaptations were copied from those exact revisions.
