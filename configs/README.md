# Configuration organization view

This directory contains the canonical project-level configuration files.
The base veRL/Hydra configuration remains packaged under verl/trainer/config.

Canonical locations:

- GRPO and multi-turn rollout: configs/grpo/
- Tool configuration: configs/tools/
- Base verl trainer configuration: verl/trainer/config/
- SFT configuration: configs/sft/ and verl/trainer/config/

The former example and top-level SFT configuration paths were removed after
the migration; use the canonical locations above.
