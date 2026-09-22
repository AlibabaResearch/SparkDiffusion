# Agent Workflow

This repository ships reusable agent skills under `.agents/skills/`.

- Use `sparkdiffusion-setup` for environment, path, dependency, and preflight work.
- Use `sparkdiffusion-finetune` for sparse finetuning.
- Use `sparkdiffusion-distill` for few-step distillation.
- Use `sparkdiffusion-inference` for T2V/I2V generation and checkpoint comparisons.

Read the relevant skill before running a GPU job. Prefer the existing launchers over hand-written commands, use repository-relative paths or environment variables, and do not modify model/data assets in the source tree.
