"""Test cohort-specific OCSVM boundaries for Task 5.

Run from the repository root. Cohorts use account age and P1-P3 activity only.
The same F0 inputs and global preprocessing are used in every OCSVM fit.
No labels, teammate Task 5 predictions, or Kaggle scores select a variant.
Pass --export to prepare the prespecified cohort-plus-trim submission candidate.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM


ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--export", action="store_true")
args = parser.parse_args()
raw = pd.read_csv(ROOT / "src/task5/data/raw/test.csv")
clean = raw.copy()
for period in range(1, 5):
    lat, lon = f"login_lat_{period}", f"login_lon_{period}"
    invalid = (raw[lat].notna() & ~raw[lat].between(-90, 90)) | (raw[lon].notna() & ~raw[lon].between(-180, 180))
    clean.loc[invalid, [lat, lon]] = np.nan
for column in ("refund_count_2", "refund_count_3", "guild_changes_2", "guild_changes_3"):
    clean.loc[raw[column].eq(-1), column] = np.nan

security = {"password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"}
coordinates = {f"login_{axis}_{period}" for period in range(1, 5) for axis in ("lat", "lon")}
feature_cols = [column for column in raw if column != "player_id"]
behavior = [column for column in feature_cols if column not in security | coordinates]
assert len(behavior) == 110

f0_frame = clean[behavior]
f0_values = RobustScaler().fit_transform(f0_frame.fillna(f0_frame.median()).to_numpy(dtype=float))
start = perf_counter()
global_model = OneClassSVM(kernel="rbf", nu=0.10, gamma="scale", cache_size=500)
global_model.fit(f0_values)
global_scores = -global_model.score_samples(f0_values)
saved = pd.read_csv(ROOT / "src/submission/kaggle_t1234_peer_t5_ocsvm_f0_20261121.csv")
global_top = set(np.argsort(-global_scores, kind="stable")[:2046])
assert global_top == set(np.flatnonzero(saved.task5.to_numpy() == 1))
print(f"Global OCSVM reproduced exactly | rows {len(raw):,} | fit seconds {perf_counter()-start:.1f}")

age = raw.account_age_days
activity = raw[[f"activity_level_{period}" for period in (1, 2, 3)]].median(axis=1)
age_cut = float(age.median())
activity_cut = float(activity.median())
cohorts = np.where(age.isna(), "age_missing",
                   np.where(age.le(age_cut), "young", "old") + "__"
                   + np.where(activity.le(activity_cut), "low_activity", "high_activity"))
cohort_names, cohort_counts = np.unique(cohorts, return_counts=True)
assert len(cohort_names) == 5 and cohort_counts.min() > 500
print("Cohorts | age median %.2f | P1-P3 activity median %.3f | %s" %
      (age_cut, activity_cut, dict(zip(cohort_names, cohort_counts))))

# Use the original full-file Tail score only to identify 5% of reference rows
# for the second OCSVM variant; its scores never enter OCSVM inference.
temporal_selected = [
    "login_count", "login_time_variance", "location_changes", "device_count",
    "ip_hash_entropy", "activity_level", "session_duration_mean",
    "play_pattern_consistency", "skill_rating", "skill_trajectory",
    "kda_mean", "reaction_time_mean", "currency_transfers",
    "purchase_amount_mean", "friend_removes", "communication_frequency",
]
temporal = pd.DataFrame(index=clean.index)
for family in temporal_selected:
    values = clean[[f"{family}_{period}" for period in range(1, 5)]].copy()
    values.columns = range(1, 5)
    earlier_median = values[[1, 2, 3]].median(axis=1)
    adjacent = pd.concat([(values[period + 1] - values[period]).abs() for period in range(1, 4)], axis=1)
    has_two = values.notna().sum(axis=1).ge(2)
    delta34 = values[4] - values[3]
    temporal[f"{family}__delta34"] = delta34
    temporal[f"{family}__abs_delta34"] = delta34.abs()
    temporal[f"{family}__baseline_shift"] = values[4] - earlier_median
    temporal[f"{family}__max_adjacent_change"] = adjacent.max(axis=1)
    temporal[f"{family}__temporal_range"] = (values.max(axis=1) - values.min(axis=1)).where(has_two)
    temporal[f"{family}__temporal_std"] = values.std(axis=1, ddof=0).where(has_two)
f1_frame = pd.concat([f0_frame, temporal], axis=1)
assert f1_frame.shape[1] == 206
f1_filled = f1_frame.fillna(f1_frame.median()).to_numpy(dtype=float)
f1_values = RobustScaler().fit_transform(f1_filled)


def empirical_tail(reference, query, top_k=8):
    ordered = np.sort(reference, axis=0)
    n_ref = len(ordered)
    surprises = np.empty(query.shape, dtype=float)
    for column in range(query.shape[1]):
        low = np.searchsorted(ordered[:, column], query[:, column], side="right")
        high = n_ref - np.searchsorted(ordered[:, column], query[:, column], side="left")
        surprises[:, column] = -np.log((np.minimum(low, high) + 1) / (n_ref + 1))
    return np.partition(surprises, -top_k, axis=1)[:, -top_k:].sum(axis=1)


tail_initial = empirical_tail(f1_values, f1_values)
retained_count = len(raw) - int(np.ceil(len(raw) * 0.05))
tail_reference = np.argsort(tail_initial, kind="stable")[:retained_count]
tail_final = empirical_tail(f1_values[tail_reference], f1_values)
tail_top = set(np.argsort(-tail_final, kind="stable")[:2046])
scored_tail = pd.read_csv(ROOT / "src/submission/kaggle_t1234_peer_t5_tail_reference_trim5pct_20261121.csv")
assert tail_top == set(np.flatnonzero(scored_tail.task5.to_numpy() == 1))
retained_mask = np.zeros(len(raw), dtype=bool)
retained_mask[tail_reference] = True
print(f"Tail 5% reference exclusion reproduced exactly | retained {retained_count:,}")


def fit_cohorts(keep_mask):
    calibrated = np.empty(len(raw), dtype=float)
    details = []
    for cohort in cohort_names:
        query_idx = np.flatnonzero(cohorts == cohort)
        fit_idx = query_idx[keep_mask[query_idx]]
        assert len(fit_idx) > 500
        model = OneClassSVM(kernel="rbf", nu=0.10, gamma="scale", cache_size=500)
        model.fit(f0_values[fit_idx])
        reference_scores = np.sort(-model.score_samples(f0_values[fit_idx]))
        query_scores = -model.score_samples(f0_values[query_idx])
        calibrated[query_idx] = np.searchsorted(reference_scores, query_scores, side="right") / len(reference_scores)
        details.append({"cohort": cohort, "query_rows": len(query_idx), "fit_rows": len(fit_idx),
                        "support_vectors": len(model.support_)})
    assert np.isfinite(calibrated).all()
    return calibrated, pd.DataFrame(details)


variants = {}
for name, keep_mask in (("cohort_all", np.ones(len(raw), dtype=bool)),
                        ("cohort_tail_trim5", retained_mask)):
    start = perf_counter()
    scores, detail = fit_cohorts(keep_mask)
    selected = set(np.argsort(-scores, kind="stable")[:2046])
    variants[name] = {"scores": scores, "selected": selected, "detail": detail}
    selected_cohorts = pd.Series(cohorts[list(selected)]).value_counts().to_dict()
    print(f"{name} | seconds {perf_counter()-start:.1f} | common global OCSVM {len(selected & global_top)} | common Tail5 {len(selected & tail_top)} | cutoff ties {np.count_nonzero(scores == np.sort(scores)[-2046])}")
    print(detail.to_string(index=False))
    print(f"selected by cohort: {selected_cohorts}")

common = len(variants["cohort_all"]["selected"] & variants["cohort_tail_trim5"]["selected"])
print(f"Cohort variant overlap | common {common} | swapped each way {2046-common}")

# Factor check: does the selection change come from separate OCSVM boundaries,
# within-cohort percentile calibration, or reference trimming?
global_trim_model = OneClassSVM(kernel="rbf", nu=0.10, gamma="scale", cache_size=500)
global_trim_model.fit(f0_values[retained_mask])
global_trim_scores = -global_trim_model.score_samples(f0_values)
global_trim_top = set(np.argsort(-global_trim_scores, kind="stable")[:2046])


def calibrate_global_by_cohort(scores, keep_mask):
    calibrated = np.empty(len(scores), dtype=float)
    for cohort in cohort_names:
        group_idx = np.flatnonzero(cohorts == cohort)
        reference_idx = group_idx[keep_mask[group_idx]]
        reference_scores = np.sort(scores[reference_idx])
        calibrated[group_idx] = np.searchsorted(reference_scores, scores[group_idx], side="right") / len(reference_scores)
    return calibrated


global_cal_top = set(np.argsort(-calibrate_global_by_cohort(global_scores, np.ones(len(raw), dtype=bool)), kind="stable")[:2046])
global_trim_cal_top = set(np.argsort(-calibrate_global_by_cohort(global_trim_scores, retained_mask), kind="stable")[:2046])
factor_rows = [
    {"selection": "global_raw", "common_global_raw": 2046, "common_cohort_trim": len(global_top & variants["cohort_tail_trim5"]["selected"])},
    {"selection": "global_trim_raw", "common_global_raw": len(global_trim_top & global_top), "common_cohort_trim": len(global_trim_top & variants["cohort_tail_trim5"]["selected"])},
    {"selection": "global_full_group_percentile", "common_global_raw": len(global_cal_top & global_top), "common_cohort_trim": len(global_cal_top & variants["cohort_tail_trim5"]["selected"])},
    {"selection": "global_trim_group_percentile", "common_global_raw": len(global_trim_cal_top & global_top), "common_cohort_trim": len(global_trim_cal_top & variants["cohort_tail_trim5"]["selected"])},
    {"selection": "cohort_full", "common_global_raw": len(variants["cohort_all"]["selected"] & global_top), "common_cohort_trim": common},
    {"selection": "cohort_trim", "common_global_raw": len(variants["cohort_tail_trim5"]["selected"] & global_top), "common_cohort_trim": 2046},
]
print(pd.DataFrame(factor_rows).to_string(index=False))

if args.export:
    peer_path = ROOT / "src/task5/analysis/reference/peer_density_auto_v1.csv"
    with peer_path.open(newline="", encoding="utf-8") as file:
        peer_rows = list(csv.reader(file))
    sample_ids = pd.read_csv(ROOT / "src/submission/sample_submission.csv", usecols=["id"], dtype={"id": str}).id.tolist()
    assert peer_rows[0] == ["id", "task1", "task2", "task3", "task4", "task5"]
    assert len(peer_rows) == len(raw) + 1
    assert [row[0] for row in peer_rows[1:]] == sample_ids == saved.id.astype(str).tolist()
    assert sum(row[5] == "1" for row in peer_rows[1:]) == 2046
    selected = variants["cohort_tail_trim5"]["selected"]
    output_rows = [peer_rows[0]] + [row[:5] + [str(int(i in selected))]
                                    for i, row in enumerate(peer_rows[1:])]
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(output_rows)
    payload = buffer.getvalue().encode("utf-8")
    name = "kaggle_t1234_peer_t5_cohort_ocsvm_tailtrim5_v1_20261121"
    output_path = ROOT / "src/submission" / f"{name}.csv"
    if output_path.exists():
        assert output_path.read_bytes() == payload, f"Existing CSV differs: {output_path}"
    else:
        with output_path.open("xb") as file:
            file.write(payload)
    card = {
        "name": name,
        "csv_sha256": hashlib.sha256(payload).hexdigest(),
        "source_reference": str(peer_path.relative_to(ROOT)),
        "source_reference_sha256": hashlib.sha256(peer_path.read_bytes()).hexdigest(),
        "cohorts": "age_missing or 2x2 account_age_days and P1-P3 median activity_level at full-file medians",
        "predictors": "110 cleaned F0 behavioral raw features, global median imputation and RobustScaler",
        "trim": "exclude the highest original F1 empirical-tail score 5% of rows from each cohort OCSVM fit only",
        "model": "OneClassSVM RBF, nu .10, gamma scale, one model per cohort",
        "calibration": "within-cohort percentile rank against retained fit-row negative score_samples",
        "task5": "top 2046 calibrated scores, 1=anomaly; stable row-order tie break",
        "tasks1_to_4": "teammate reference text preserved exactly row by row",
        "caveat": "No Task 5 labels, F3, or Kaggle score; teammate predictions unused in fitting or candidate choice",
    }
    card_path = output_path.with_suffix(".json")
    card_payload = (json.dumps(card, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if card_path.exists():
        assert card_path.read_bytes() == card_payload, f"Existing card differs: {card_path}"
    else:
        with card_path.open("xb") as file:
            file.write(card_payload)
    print(f"Exported {output_path.relative_to(ROOT)} | SHA-256 {card['csv_sha256']}")
