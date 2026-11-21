# Task 4 Report: Game Title Detection

The Thai version is the master copy: [task4-report.th.md](task4-report.th.md). Both versions use the same section numbers.

## 1. Problem

Predict which game each 230×120 screenshot comes from. There are 5 classes, `7-11k`, `fiveN`, `freefried`, `pube` and `roblock`, scored with macro F1, which weights every class equally. Task 4 is the only task where pretrained models are allowed.

Every score in this report is macro F1 on our own validation splits, not a Kaggle score, because Kaggle shows only the combined score of all 5 tasks.

## 2. Steps

### Step 1 Explore the data

Notebook: `src/task4/eda/build_eda_original.ipynb`

| class | train images | share |
| --- | ---: | ---: |
| 7-11k | 12,429 | 39.4% |
| pube | 5,926 | 18.8% |
| freefried | 5,596 | 17.7% |
| fiveN | 4,407 | 14.0% |
| roblock | 3,188 | 10.1% |

- 31,546 train and 25,889 test images, all 230×120, none missing.
- No repeated images in train (by file SHA-256) and none shared between train and test.
- Test images are slightly darker than train: mean brightness 0.414 against 0.451.
- The classes are imbalanced. 7-11k has almost 4 times as many images as roblock, but macro F1 weights them equally.

The first dataset, `eda_original_v1`, stores each image shrunk to 16×8 RGB pixels (384 values) as a baseline.

### Step 2 Extract image features

Notebook: `src/task4/eda/build_eda_1.ipynb`

Classical features, 1,228 per image:

| feature group | count | captures |
| --- | ---: | --- |
| 16×8 RGB pixels | 384 | shrunken image |
| R, G, B mean/std | 6 | overall colour |
| colour means on a 2×4 grid | 24 | colour by screen position |
| H, S, V histograms | 48 | colour distribution |
| HOG | 756 | edge directions |
| LBP | 10 | texture |

No feature has missing values or a constant column. Class colour means are too close for colour alone to tell the games apart.

### Step 3 First model round (classical machine learning)

Notebook: `src/task4/models/build_model.ipynb`

A stratified 20% validation split keeps images with the same hash on one side. Seven models were compared (LightGBM in two sizes, 15 and 31 leaves), each on raw pixels and on the Step 2 features, then hyperparameters were tuned with 3 folds on 9,000 images.

| model | pixels | features |
| --- | ---: | ---: |
| Majority (always predicts the biggest class) | 0.11305 | |
| Logistic regression | 0.41192 | 0.72128 |
| Linear SVM (SGD) | 0.33276 | 0.67859 |
| ExtraTrees | 0.52570 | 0.61219 |
| HistGradientBoosting (tuned) | 0.57489 | 0.75287 |
| LightGBM 15 leaves (tuned) | 0.54642 | 0.74278 |
| **LightGBM 31 leaves (tuned)** | 0.58965 | **0.77164** |

- The Step 2 features beat raw pixels for every model.
- The best score is only 0.77, and roblock F1 is about 0.68.

### Step 4 test.csv changes

Notebooks: `src/task4/eda/build_eda_original_v2.ipynb`, `build_eda_1_v2.ipynb`

The instructor changed only `test.csv`; train images and labels stayed byte-identical. The image hashes were checked and the existing features were re-keyed to the new IDs (`eda_original_v2`, `eda_1_v2`) without extracting them again.

### Step 5 Switch to a pretrained ResNet18

Notebook: `src/task4/models/build_model_v2.ipynb`

Each image is resized to 112×224, which keeps its aspect ratio, padded with black above and below to 224×224, and normalized with ImageNet mean/std. Every run starts from the same weights with AdamW lr 1e-4, batch 32, seed 42 and class-weighted loss, and keeps the best epoch on validation (stopping after 3 epochs without improvement).

One change at a time on the same fold, then confirmed on a second fold:

| method | fold 1 | fold 2 |
| --- | ---: | ---: |
| LightGBM from Step 3 | 0.77164 | 0.77532 |
| ResNet18 with a frozen backbone, last layer trained | 0.70728 | |
| Full fine-tune, no augmentation | 0.94637 | 0.94962 |
| **Full fine-tune + geometry augmentation** | **0.95246** | **0.95528** |
| Full fine-tune + geometry + blur/noise | 0.95533 | 0.94346 |
| Full fine-tune + ±180° rotation | 0.94936 | |

Geometry augmentation means random rotation up to ±50°, perspective distortion and scaling by 0.8–1.18, applied to train images only.

- Fine-tuning the whole network is what moves the score from 0.77 to 0.95.
- Geometry beats no augmentation on both folds (+0.006 each).
- Blur/noise helps on fold 1 but loses badly on fold 2, so it was not chosen.
- Geometry on fold 3 as a confirmation gives 0.95568, for a 3-fold mean of **0.95447**. Fold-3 F1 per class: 7-11k 0.975, fiveN 0.952, freefried 0.969, pube 0.962, roblock 0.919.

### Step 6 Train on all train images and first submission

The geometry recipe was trained on all 31,546 train images for 7 epochs (the median best epoch across the 3 folds: 7, 8, 7), giving `t4_resnet18_geometry_full_v2` and `src/task4/submission/t4_resnet18_geometry_full_v2.csv`.

### Step 7 Follow-up: a fairer experiment setup

Notebook: `Kaggle-Final/of_friends/Task4_gpu_followup.ipynb` (outside the repo; runs on its own on a Colab/Kaggle GPU)

Step 5 picked epochs on validation and picked the recipe from the same folds it reported. This round changes the rules:

- The epoch count of every recipe is fixed before training and the last epoch is always used. Validation never picks a checkpoint.
- The old 7-epoch recipe is retrained on the same folds as a baseline, so comparisons are direct.
- The selection rule is fixed in advance: beat the baseline by at least 0.001 on average and on at least 4 of 5 folds.
- Class weights in the loss come from each fold's own train part.

Before splitting, duplicates were checked again. A pair counts as a near-duplicate only if pHash distance is at most 4, dHash distance is at most 4 and the mean thumbnail difference is at most 0.02. That finds 4 pairs, none with conflicting labels. The 5 folds come from `StratifiedGroupKFold` (seed 42), with each pair kept in one fold.

### Step 8 Compare 5 recipes

All recipes use the Step 5 preprocessing and geometry augmentation.

| recipe | model | epochs | backbone lr | head lr | schedule | status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `friend_resnet18_fixed7` (baseline) | ResNet18 | 7 | 1e-4 | 1e-4 | constant | all 5 folds |
| `resnet18_cosine18` | ResNet18 | 18 | 1e-4 | 1e-4 | cosine, 2 warmup epochs | all 5 folds |
| `resnet18_head_lr18` | ResNet18 | 18 | 5e-5 | 5e-4 | cosine, 2 warmup epochs | all 5 folds |
| `efficientnet_b0_cosine18` | EfficientNet-B0 | 18 | 1e-4 | 3e-4 | cosine, 2 warmup epochs | stopped during fold 5 |
| `convnext_tiny_cosine18` | ConvNeXt-Tiny | 18 | 5e-5 | 3e-4 | cosine, 2 warmup epochs | not run |

A cosine schedule raises the learning rate over the first 2 epochs, then lowers it along a curve to almost zero by the last epoch.

5-fold means:

| recipe | no TTA | TTA | vs baseline (TTA) |
| --- | ---: | ---: | ---: |
| `friend_resnet18_fixed7` | 0.94619 | 0.94897 | +0.0028 |
| **`resnet18_cosine18`** | 0.97369 | **0.97551** | **+0.0293** |
| `resnet18_head_lr18` | 0.97025 | 0.97243 | +0.0262 |

Per fold (no TTA / TTA):

| fold | fixed7 | cosine18 | head_lr18 |
| ---: | --- | --- | --- |
| 1 | 0.94216 / 0.94457 | 0.97287 / 0.97370 | 0.97313 / 0.97384 |
| 2 | 0.94183 / 0.94578 | 0.97469 / 0.97788 | 0.97175 / 0.97341 |
| 3 | 0.95086 / 0.95615 | 0.97162 / 0.97528 | 0.96882 / 0.97117 |
| 4 | 0.94528 / 0.94602 | 0.97547 / 0.97502 | 0.96841 / 0.97296 |
| 5 | 0.95083 / 0.95230 | 0.97381 / 0.97567 | 0.96914 / 0.97076 |

- The top recipe that passes the rule is `resnet18_cosine18` + TTA, better than the baseline on every fold.
- Same model, only trained longer with a cosine schedule, gains almost 0.03.
- The baseline scores 0.946 here rather than 0.954 as in Step 5 because the folds differ and validation no longer picks the epoch.
- EfficientNet-B0 has results for some folds before it stopped, for example fold 3 at 0.9764 / 0.9793 and fold 4 at 0.9799 / 0.9796. Those are above cosine18 on the same folds, but without all 5 folds it was not part of the selection.

### Step 9 Test-time augmentation (TTA)

At prediction time the model sees each image five ways: original, rotated −8° and +8°, scaled 0.95 and 1.05. The probabilities are averaged before choosing a class. No retraining is needed.

| fold | no TTA | TTA | change |
| ---: | ---: | ---: | ---: |
| 1 | 0.97287 | 0.97370 | +0.0008 |
| 2 | 0.97469 | 0.97788 | +0.0032 |
| 3 | 0.97162 | 0.97528 | +0.0037 |
| 4 | 0.97547 | 0.97502 | −0.0005 |
| 5 | 0.97381 | 0.97567 | +0.0019 |

TTA adds +0.0018 on average, helps on 4 of 5 folds, and every recipe gains from it. On test, TTA changes 1,249 of 25,889 predictions (4.8%).

### Step 10 Error analysis

Out-of-fold predictions of `resnet18_cosine18` + TTA on all 31,546 images.

| class | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| 7-11k | 0.98 | 0.99 | 0.99 |
| fiveN | 0.98 | 0.97 | 0.97 |
| freefried | 0.98 | 0.98 | 0.98 |
| pube | 0.98 | 0.97 | 0.98 |
| roblock | 0.96 | 0.96 | 0.96 |

Overall accuracy is 0.98. roblock is still the weakest class, but it rose from 0.92 in Step 5 to 0.96.

Error rate by image property, split into 4 equal-sized bands (Q1 = lowest value):

| image property | Q1 | Q2 | Q3 | Q4 |
| --- | ---: | ---: | ---: | ---: |
| brightness | 2.9% | 1.3% | 1.6% | 2.6% |
| contrast | 3.2% | 1.7% | 1.8% | 1.6% |
| amount of edges/detail (edge energy) | 4.0% | 1.9% | 1.4% | 1.1% |

- Images with little detail are the hardest, with almost 4 times the error rate of the most detailed ones.
- Low contrast nearly doubles the error rate.
- Brightness hurts at both ends: very dark and very bright images.
- The model is also less confident on these groups (mean confidence 0.97 against 0.99).

### Step 11 Train on all train images and build the submission

- `resnet18_cosine18` was trained on all 31,546 train images for 18 epochs with the same recipe. Loss fell from 0.529 in the first epoch to 0.0119 in the last. There is no validation at this step; the reference score is the CV from Step 8.
- Test predicted with TTA gives 7-11k 5,017, fiveN 5,579, freefried 3,764, pube 7,247, roblock 4,282. This task4 column is in `kaggle_t12345_081771.csv`, which scores **0.81771** overall, the team's best so far.
- The checkpoint was exported as `resnet18_cosine18_pretrained.pt` (42.7 MB), holding the weights, preprocessing values and training recipe together.
- The predicted class mix on test differs from train: 7-11k is 39% of train but 19% of test predictions, and pube is 19% of train but 28% of test predictions. It is unknown whether test really has a different mix or the model is biased.

### Step 12 Check the results on a second machine

Run on an RTX 3050, separate from the training machine.

- **Checkpoint:** loading `resnet18_cosine18_pretrained.pt` and predicting test with the same TTA matches the task4 column of the 0.81771 file on 25,883 of 25,889 rows (99.98%). The other 6 rows are probably floating-point differences between machines.
- **Step 5:** rerunning folds 1 and 2 gives 0.95246 and 0.95528, identical to every digit.
- **Near-duplicates:** the Step 7 checks are very strict, so looser thresholds were tried (pHash at most 4, dHash at most 16, chosen by viewing image pairs). They group 1,806 images, almost all 7-11k. On fold 1 of Step 5, validation images with a partner in train are indeed predicted better, but they inflate the score by only 0.00076, which doesn't change any conclusion (`src/task4/eda/build_eda_near_duplicate_v3.ipynb`).
- **TTA:** TTA tested separately on the Step 5 recipe with slightly stronger views (±10°, 0.9/1.1) gives +0.0031 and +0.0018 on folds 1 and 2, the same direction as Step 9 (`src/task4/models/build_model_v3.ipynb`).
- Four roblock–7-11k pairs pass the thresholds but carry different labels (`PLY11408`–`PLY02786`, `PLY11408`–`PLY19088`, `PLY03613`–`PLY04719`, `PLY06207`–`PLY30885`). They have not been viewed yet.

## 3. Results summary

| step | method | macro F1 |
| --- | --- | ---: |
| 3 | LightGBM on classical features | 0.772 (holdout) |
| 5 | ResNet18 fine-tune + geometry | 0.954 (3-fold mean) |
| 8–9 | ResNet18 cosine 18 epochs + TTA | **0.976** (5-fold mean) |

Each row comes from a different validation setup, so the table shows overall progress only. The fair comparison is Step 8, where every recipe uses the same folds.

What helped most, in order: fine-tuning the whole pretrained CNN (+0.18), longer training with a cosine schedule (+0.03), geometry augmentation (+0.006) and TTA (+0.002).

## 4. Limitations

- The final recipe was chosen from the same out-of-fold predictions that give the 0.9755, so that number may be slightly high. The +0.029 gap is far larger than the fold-to-fold spread, though.
- The folds catch few near-duplicates, so CV may be inflated by about 0.001.
- EfficientNet-B0 has not finished all 5 folds, and ConvNeXt-Tiny has not been run.
- Kaggle shows only the combined score, so Task 4's effect on the leaderboard can't be seen on its own.

## 5. Next steps

1. Finish EfficientNet-B0 on all 5 folds and try the ensemble with `resnet18_cosine18` that the follow-up notebook already defines.
2. Target low-contrast, low-detail, and very dark or bright images, for example with random brightness/contrast changes during training, measured on the same folds first.
3. View the four roblock–7-11k pairs whose labels disagree.

## 6. Files

| step | files |
| --- | --- |
| 1 | `src/task4/eda/build_eda_original.ipynb` |
| 2 | `src/task4/eda/build_eda_1.ipynb`, `src/task4/image_features.py` |
| 3 | `src/task4/models/build_model.ipynb` |
| 4 | `src/task4/eda/build_eda_original_v2.ipynb`, `src/task4/eda/build_eda_1_v2.ipynb` |
| 5–6 | `src/task4/models/build_model_v2.ipynb`, `src/task4/models/image_training.py` |
| 7–11 | `Kaggle-Final/of_friends/Task4_gpu_followup.ipynb`, `Kaggle-Final/of_friends/resnet18_cosine18_pretrained.pt` (outside the repo) |
| 11 | `Kaggle-Final/kaggle_t12345_081771.csv` (outside the repo) |
| 12 | `src/task4/eda/build_eda_near_duplicate_v3.ipynb`, `src/task4/models/build_model_v3.ipynb`, `src/task4/models/tta.py` |
