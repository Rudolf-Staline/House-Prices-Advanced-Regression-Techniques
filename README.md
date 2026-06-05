# engineering-profile-90-days

This repository documents a 90-day program to build a strong generalist engineering profile through practical projects, structured learning, and consistent reporting.

The program focuses on four complementary tracks:

- **Machine Learning and Data Science**: clean, reproducible projects with production-minded structure.
- **Engineering Projects**: readable code, automation, documentation, and GitHub-ready deliverables.
- **Reading and Thinking**: short notes from books and technical resources.
- **English and Communication**: regular practice through summaries, project write-ups, and reports.

## Repository structure

```text
engineering-profile-90-days/
├── projects/   # Portfolio projects
├── reading/    # Reading notes
├── language/   # English practice and writing exercises
├── reports/    # Weekly progress reports
└── ROADMAP.md  # 90-day learning roadmap
```

## Current project

The first portfolio project is a reproducible baseline for the Kaggle competition **House Prices — Advanced Regression Techniques**.

Project folder:

```text
projects/kaggle-house-prices/
```

It includes:

- exploratory data analysis notebook;
- reusable preprocessing code;
- Ridge and RandomForest baselines;
- cross-validation using Kaggle's log-RMSE objective;
- a script to generate a Kaggle-compatible submission file.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then follow the project-specific instructions in [`projects/kaggle-house-prices/README.md`](projects/kaggle-house-prices/README.md).
