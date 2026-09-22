# Claude Code Workflow

Project skills are available under `.claude/skills/`; their canonical workflows are
maintained under `.agents/skills/`. Before acting, load the skill matching the task:

- `.claude/skills/sparkdiffusion-setup/SKILL.md`
- `.claude/skills/sparkdiffusion-finetune/SKILL.md`
- `.claude/skills/sparkdiffusion-distill/SKILL.md`
- `.claude/skills/sparkdiffusion-inference/SKILL.md`

Use the repository launchers, preserve relative-path conventions, validate paths before GPU jobs, and keep generated assets outside source directories.
