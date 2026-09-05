# TravelGRPO source package

This is the canonical home of the project-owned collection, evaluation, and
SFT helpers. It is installed as the `travel_grpo` package and can also be used
directly from a checkout with `PYTHONPATH=src`.

For an installed checkout use `python -m pip install -e .`; when using the
package without installation, export both `src/` and
`environments/TravelGym/` on `PYTHONPATH`.

The physical implementation is organized as follows:

- `collection/`: Teacher collection, trajectory cleaning, task pools, SFT
  corpus construction, and data-preparation utilities;
- `evaluation/`: HTTP evaluation, native-runtime preflight, fixed-manifest
  construction, final200 planning, result analysis, and the TravelGym public
  contract;
- `training/sft/`: Qwen3.5 token masks, task-level SFT splitting, and
  canonical-corpus validation;
- `environment/`: the boundary map for the separately packaged TravelGym
  environment and veRL tools;
- `training/grpo/`: the boundary map for the veRL GRPO/SGLang runtime, which
  intentionally remains under `verl/`.

The historical top-level implementation paths were removed. Use the module
paths above so the repository has one physical copy of each project-owned
implementation.
