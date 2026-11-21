# CPE342 Machine Learning — Player Intelligence System

**Kaggle Competition | Karena Thailand Player Intelligence Competition**

This repository is our team's workspace for the **CPE342 Machine Learning** project. We are building an end-to-end **Player Intelligence System** for a fictional gaming platform, Karena Thailand, across five machine learning tasks: cheat detection, player segmentation, spending prediction, game screenshot classification, and account security monitoring.

## Repository Structure

```text
.
├── final-notebook/          # Final submission notebooks (one notebook per task)
│   ├── task1.ipynb
│   ├── task2.ipynb
│   ├── task3.ipynb
│   ├── task4.ipynb
│   └── task5.ipynb
│
├── Jakkarin/                # Personal experiments and notes
├── JoJiLLst/
├── R1NNE0/
├── Thanawat/
└── Thawat/
```

### `final-notebook/` — Final Submission

This directory contains the **five final notebooks** submitted for evaluation. Each notebook should be **self-contained and reproducible**, covering the complete ML workflow for its task:

1. Load data and perform exploratory analysis (EDA).
2. Clean and preprocess data; engineer features when useful.
3. Train and validate the model using the task's evaluation metric.
4. Report results and key observations.
5. Run inference and produce the required submission output.

> **Goal:** A teammate should be able to run each notebook from top to bottom without relying on files from anyone's personal workspace. Document any required external datasets, packages, or paths.

### Personal Workspaces — Experiments & Collaboration

Each member has their own folder to share **experiments, notebooks, ideas, and findings**. There is **no required folder structure or file naming convention** inside these folders. Feel free to organize your workspace however you prefer.

**Everyone can push work directly to their own shared folder.** When an experiment improves a task, share the relevant insights with the team and integrate the best reproducible version into `final-notebook/`.

## Competition Tasks

| Task                                  | Objective                                                     | ML Type                        | Evaluation Metric         | Final Notebook |
| ------------------------------------- | ------------------------------------------------------------- | ------------------------------ | ------------------------- | -------------- |
| **1 — Anti-Cheat Pre-Filter**         | Identify true cheaters among flagged accounts                 | Binary classification          | **Macro F₂**              | `task1.ipynb`  |
| **2 — Player Segment Classification** | Classify players into four behavioral segments                | Multi-class classification     | **Macro F₁**              | `task2.ipynb`  |
| **3 — Monthly Spending Prediction**   | Predict player spending in THB over the next 30 days          | Regression                     | **Normalized MAE (NMAE)** | `task3.ipynb`  |
| **4 — Game Title Detection**          | Identify games from screenshots                               | Image classification           | **Macro F₁**              | `task4.ipynb`  |
| **5 — Account Security Monitoring**   | Detect unusual account behavior without labeled training data | Unsupervised anomaly detection | **Macro F₃**              | `task5.ipynb`  |

### Task 1 — Anti-Cheat Pre-Filter

Build a binary classifier to distinguish **true cheaters** from **false positives** among accounts flagged by the monitoring system. The challenge is to improve review efficiency while avoiding unnecessary action against legitimate, skilled players.

- **Problem type:** Binary classification
- **Metric:** Macro F₂ score

### Task 2 — Player Segment Classification

Classify players into four behavioral segments to support more targeted marketing and player engagement.

| Class | Segment             | Description                                   |
| ----- | ------------------- | --------------------------------------------- |
| `0`   | Casual Player       | Relaxed gameplay and low spending             |
| `1`   | Competitive Grinder | High playtime and ranked/competitive focus    |
| `2`   | Social Player       | Friend-focused gameplay and cosmetic spending |
| `3`   | Whale               | High spending and status-driven behavior      |

- **Problem type:** Multi-class classification
- **Metric:** Macro F₁ score

### Task 3 — Monthly Spending Prediction

Predict **how much each player will spend (THB) over the next 30 days**. These predictions can support revenue forecasting, VIP prioritization, and smarter promotions.

- **Problem type:** Regression
- **Target:** Next 30-day spending in Thai baht (THB)
- **Metric:** Normalized Mean Absolute Error (NMAE)

### Task 4 — Game Title Detection

Classify game screenshots so that customer support tickets and campaign submissions can be routed or verified automatically.

| Class | Game            |
| ----- | --------------- |
| `0`   | 7-Eleven Knight |
| `1`   | FiveN           |
| `2`   | Free Fried      |
| `3`   | PUBE Mobile     |
| `4`   | Roblock         |

- **Problem type:** Image classification
- **Metric:** Macro F₁ score

### Task 5 — Account Security Monitoring

Build an **unsupervised anomaly detection** system to identify suspicious account behavior, such as possible account takeovers, bot activity, or unusual region-related usage patterns.

- **Problem type:** Unsupervised anomaly detection
- **Metric:** Macro F₃ score

## Team Workflow

1. **Experiment** in your personal folder. Try different preprocessing strategies, features, models, and hyperparameters.
2. **Document** your findings, including validation scores, assumptions, and what did or did not work.
3. **Share** useful notebooks and insights by pushing them to the repository.
4. **Consolidate** the strongest approach for each task in its corresponding `final-notebook/taskX.ipynb`.
5. **Verify** that the final notebook runs from a clean start and produces the expected predictions or submission file.

## Repository Rules

**Do not commit large files or generated artifacts to GitHub.** In particular, avoid pushing:

- Raw or processed datasets (`.csv`, `.parquet`, etc.).
- Trained model files (`.joblib`, `.pkl`, `.pt`, `.pth`, etc.).
- Large generated predictions, checkpoints, caches, or notebook outputs.

Instead, keep the code needed to **regenerate these files** in your notebooks. If a large file must be shared, use a separate file-sharing method and document how teammates can access it. Consider adding these file patterns to `.gitignore` and clearing unnecessary notebook outputs before committing.

## Final Submission Checklist

- [ ] All five notebooks are in `final-notebook/` with the correct filenames.
- [ ] Every notebook contains the full ML workflow and runs top to bottom.
- [ ] Validation uses the correct metric for its task.
- [ ] Required data paths, packages, and setup steps are documented.
- [ ] Inference and submission generation are included.
- [ ] No large datasets, model binaries, or temporary artifacts are committed.

---

**CPE342 Machine Learning · Karena Thailand Player Intelligence Competition**
