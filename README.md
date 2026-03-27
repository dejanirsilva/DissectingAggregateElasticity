# Dissecting Aggregate Elasticity

This repository is structured to keep paper files, analysis code, notes, and submission materials in one place while syncing only the paper project with Overleaf.

The current manuscript was imported from Overleaf on March 26, 2026. During that import cleanup, manuscript files were kept under `paper/`, supporting notes were moved to `Notes/`, archived drafts were moved to `Notes/archive/`, and code/scripts that had been living inside the Overleaf project were moved into `src/`.

## Repository layout

- `paper/`: main paper source synced with Overleaf
- `Notes/`: derivations, scratch notes, and supporting writeups
- `src/`: data work, model code, replication scripts, and figure generation
- `submissions/`: journal-specific packages, letters, and reports
- `literature/`: local reference material not intended for Overleaf
- `utils/`: shared scripts, bibliography files, and LaTeX styles

## Overleaf workflow

1. Add the Overleaf git remote.
2. Pull the existing Overleaf project into `paper/`.
3. Keep editing locally or in Overleaf and sync with the helper scripts in `utils/scripts/`.

See `OVERLEAF.md` for the exact commands.
