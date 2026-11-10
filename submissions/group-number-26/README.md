# Group Number 26 competition submission

Kaggle team: Group Number 26. Contributors: Kevin Bradicich (style-feature KNN and validation comparison), [partner name and NetID] (word TF-IDF logistic regression), and [partner name and NetID] (word and character TF-IDF linear SVM). Confirm the team roster and NetIDs before opening the course PR.

The selected model is uniform-vote KNN with 15 neighbors. It extracts 23 style features, fits a standard scaler on labeled training data, and uses Euclidean distance. Its validation AUROC was 0.9748 on the shared 2,000-passage holdout.

## Setup

Use Python 3.13 and run from the repository root. Place the course `train.csv`, `unlabeled.csv`, and `test.csv` files in `data/`.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r submissions/group-number-26/requirements.txt
```

## Reproduce validation

Use the team’s saved `data/group26_validation_ids.txt`, or generate it if absent:

```sh
python submissions/group-number-26/team_knn.py --make-split
```

The split uses seed `20261023` and contains 1,000 human and 1,000 AI passages. Its SHA-256 is `7aa7648f5f334cc462dd33918433f284c7e3f0af2283bcbff9b898f8268dee8c`. The same seed controls the nested training subsets.

```sh
python submissions/group-number-26/team_knn.py --out data/group26_final_results
```

This writes `metrics.json` and `validation_scores.csv` with the KNN sweep, matched logistic control, learning curve, unlabeled-data scaler comparison, feature ablation, and validation error analysis. Add `--partner-logistic PATH` and `--partner-svm PATH` to score the team’s private prediction exports on the same IDs. The runner verifies their IDs and labels. Those exports reproduce the comparison metrics. Retraining the two partner models requires their separate code.

The earlier course-method comparison is available with `python submissions/group-number-26/course_models.py --out data/group26_course_results`.

## Generate Kaggle predictions

```sh
python submissions/group-number-26/team_knn.py --submission data/group26_knn_submission.csv
```

This fits the selected configuration on all 10,000 labeled passages and writes the `id,label` CSV locally. Choose fresh output paths when rerunning either command. Keep course data, validation exports, and predictions out of Git.

## AI assistance

Codex helped discuss model and feature choices, implement and run experiments, check the partner score joins, inspect errors, and draft the report. The final workflow checked that the validation IDs and labels match the shared split and that the local prediction file contains the expected 3,000 test IDs. Add each teammate’s AI use and independent verification before opening the course PR.
