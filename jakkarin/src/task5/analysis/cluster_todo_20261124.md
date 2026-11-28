# Task 5 cluster analysis — working to-do list (24 November)

Living checklist, not a final plan. Items may be replaced as results come in. Findings go to
`cluster_insights_20261124.md` in this folder.

## Ground rules
- No Kaggle submissions; prepare candidate CSVs only (write-once names).
- Main targets: own best (RATIO_EXPERT, 0.81230) and best-known (0.81771). Other models describe how group choices differ.
- Moves are decided per cluster with a stated rule, never by picking individual rows.
- Public scores of submitted models are used for analysis only; note the leaderboard-overfitting risk.

## To do
- [x] A. Inventory: 13 distinct 2,046-alert models (Jaccard < 0.9), candidate pool = union (7,609 players).
- [x] B1. Feature spaces: quantile-normal family levels + P4 jumps + security flags; PCA version; mechanism space (ICA sources, ratio score) for interpretation.
- [x] B2. K-means on the pool, k = 2..30, silhouette per k; keep every k near the best; seed stability (ARI).
- [x] B3. Cluster profiles: size, shifted features, flags, expert populations, null excess (real vs block-shuffled null), ratio score, support count.
- [x] C1. Proximity: centroid distances, Ward merge tree on centroids, super-groups.
- [x] C2. Nesting across k: how clusters split as k grows.
- [x] D1. Model x cluster coverage matrix and each model's composition.
- [x] D2. RATIO_EXPERT vs best-known per-cluster differences.
- [x] D3. Score relationship: per-cluster coverage vs public score across submitted models; consecutive-submission swaps by cluster.
- [x] D4. Per-cluster marginal value estimate (regularized, super-group level).
- [x] E. Move proposals (in/out) for RATIO_EXPERT and best-known, with label-free evidence; build top candidate CSVs.
- [x] F. Extra angles: other clusterers (GMM, HDBSCAN), clustering the disagreement set, mechanism-space clustering, outside-pool groups.
- [x] G. Summary: grouping ideas, what each model does, which moves raised scores, in/out tendencies, ordered list of experiments to submit.

## Status (end of overnight run)
- Done. Findings: `cluster_insights_20261124.md`. Figures: `cluster_types_by_model_20261124.png`, `moderate_cluster_depth_20261124.png`.
- Overrides made while working: the quantile transform was replaced by average-rank normal scores (tie artifact); the cluster null was supplemented by the Round 12 asymmetry rule (blind to single-family mechanisms); C2 nesting was replaced by robustness across k and seeds; D4 per-cluster values were replaced by type-level values because per-cluster estimates were unstable.
- Five candidate CSVs are prepared (not submitted); the order to test them is in Section 7 of the insights file.
- Not done: GMM and mechanism-space (ICA) clustering, and clustering of players outside the pool. Earlier residual-dependence and k-NN tests already found no excess outside the nominations, so these were given lower priority. The analysis scripts live only in the Claude scratchpad and still need to be ported into `report.ipynb`.
