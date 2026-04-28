# CLAUDE.md

This is a **Snakemake-based physics analysis pipeline** for measuring Lambda hyperon directed flow (v1) in heavy-ion collisions at RHIC/BES energies (7.7–27 GeV).

## Environment

```bash
conda activate lambda_v1
# create/update: sh update_env.sh
```

## CERN ROOT (Docker)

ROOT is not installed locally. Use the Docker image for any ROOT commands:

```bash
docker run --rm -v "$(pwd)":/work -w /work rootproject/root:latest \
  root -l -b -q 'script.cpp("args")'
```

Image: `rootproject/root:latest` (already pulled).

## Running

```bash
snakemake -n                   # dry run (always do this first)
snakemake --cores all          # full pipeline → plots/paper/report.pdf
snakemake --cores all <target> # specific target
sh create_dag.sh               # visualize DAG
```

## Rules

- Always dry-run before executing.
- Never run steps involving `fit_v1.py` or `fit_v1_pt.py` without explicit confirmation.
- Never delete or overwrite output ROOT files without explicit instruction.
- Never modify `config.yaml` without confirmation.
- Never run grid/cluster jobs without being asked.
- For diagnostics, use sys_tag_0; place figures in the correct directory.
- If physics intent is unclear, ask before implementing.
- If a change affects more than one rule, summarize the blast radius and confirm.
- Never silently skip a step — if something can't be done, say so explicitly.
- Prefer small, focused commits.

## Code Style

- Python: PEP8, type hints for new functions, docstrings for non-obvious functions.
- No magic numbers — define constants in `config.yaml`.
- Don't refactor working code unless asked — fix only the specific thing requested.
- Use `uproot` (not PyROOT), `pathlib.Path` (not `os.path`), `subprocess.run(..., check=True)` (not `os.system()`).
- Run `git status` before committing.

## Architecture Reference

See @CLAUDE_architecture.md for pipeline flow, scripts reference, data layout, and config parameters.
