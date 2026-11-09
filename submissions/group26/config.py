"""Shared paths, seeds and model constants for the Group 26 pipeline.

Every path here is relative to this directory, so the scripts run from the
repository root as `python submissions/group26/<script>.py` or from inside
`submissions/group26/` directly. Nothing outside this folder is read or
written, and the competition CSVs are never committed.

Data location
-------------
Put the Kaggle CSVs in `submissions/group26/data/`. `find_data_dir()` also
accepts the nested folder the Kaggle download produces, e.g.
`data/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/`, so the archive can
be unzipped in place. Override with the `GROUP26_DATA_DIR` environment
variable or by passing an explicit path on the command line.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Locations --------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_DIR / "data"
RESULTS_DIR = PACKAGE_DIR / "results"
DEFAULT_SUBMISSION = PACKAGE_DIR / "submission.csv"

#: Validation IDs shared across the team. Authoritative definition of the
#: 2,000-row holdout; see SPLIT_SEED below for how it was generated.
VALIDATION_IDS = PACKAGE_DIR / "group26_validation_ids.txt"

EXPECTED_FILES = (
    "train.csv",
    "unlabeled.csv",
    "test.csv",
    "train_starter_2000.csv",
    "sample_submission.csv",
)

#: The subset of EXPECTED_FILES the core pipeline cannot run without.
REQUIRED_FILES = ("train.csv", "test.csv")


# --- Seeds ------------------------------------------------------------------

#: Seed for the stratified 80/20 validation split.
#:
#: Reproducing `group26_validation_ids.txt` requires sorting the train.csv rows
#: by the STRING form of the id before splitting, so "10" sorts before "100".
#: Splitting the original CSV order or a numeric id order with the same seed
#: gives only chance overlap (406/2000). With the string sort it regenerates
#: the file byte for byte.
SPLIT_SEED = 20261023

#: Size of the held-out validation set, in rows.
VALIDATION_ROWS = 2000

#: Expected dataset shape; asserted by the loaders.
LABELED_ROWS = 10_000
TEST_ROWS = 3000


# --- Selected model ---------------------------------------------------------
# Locally weighted logistic regression over 23 stylometric features.
# These constants are the configuration selected on the shared validation
# split; the scripts that measure them are added in later milestones.

N_NEIGHBORS = 75
LOCAL_C = 1.0

#: Clip each standardized feature to this interval before neighbour search and
#: the local fit. Measured as an improvement over the unclipped model.
CLIP_LIMIT = 5.0

#: Baseline comparison model, for the model-comparison section.
KNN_K_GRID = (15, 35, 75)


def find_data_dir(root: Path | None = None) -> Path:
    """Return the directory that actually holds the competition CSVs.

    Looks in `data/` first, then in any single sub-directory of it, so an
    unzipped Kaggle download works without being flattened by hand.
    """
    if root is None:
        env = os.environ.get("GROUP26_DATA_DIR")
        root = Path(env).expanduser() if env else DATA_ROOT

    root = Path(root)
    candidates = [root, *sorted(p for p in root.glob("*") if p.is_dir())]
    for candidate in candidates:
        if all((candidate / name).exists() for name in REQUIRED_FILES):
            return candidate

    searched = "\n  ".join(str(c) for c in candidates) or f"  {root}"
    raise FileNotFoundError(
        "Could not find the competition CSVs "
        f"({', '.join(REQUIRED_FILES)}). Looked in:\n  {searched}\n"
        "Download them from Kaggle into submissions/group26/data/, or set "
        "GROUP26_DATA_DIR."
    )


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
