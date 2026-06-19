# Manuscript

`main.tex` + `refs.bib` — *Inferring the Potentially-Hazardous Status of NEOs from
Observed Close-Approach Kinematics: A Selection-Aware, Leakage-Controlled Study*.

Figures are pulled from `../results/figures/` (via `\graphicspath`). Regenerate them
and the cross-validation results table first:

```bash
python ../.claude/skills/run-neos-analysis/driver.py --paper
```

Then compile (requires a LaTeX toolchain — not present in the dev container, so the
PDF has not been built here):

```bash
latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main x2
```

Status: text + bibliography drafted; structure validated (balanced environments,
all referenced figures present, all citations resolve). Pending: author
contributions paragraph, target-journal style file, and the actual PDF build.
