"""Build one missing-aware Empirical Tail candidate with a 5% trimmed ECDF reference.

Run from the repository root. Invalid values become missing, and each feature's
ECDF uses only observed values. All supplied accounts remain eligible for alerts.
This candidate has no known Task 5 F3.
"""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[3]
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
features = [column for column in raw if column != "player_id"]
behavior = [column for column in features if column not in security | coordinates]
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
    prior_median = values[[1, 2, 3]].median(axis=1)
    adjacent = pd.concat([(values[period + 1] - values[period]).abs() for period in range(1, 4)], axis=1)
    at_least_two = values.notna().sum(axis=1).ge(2)
    delta34 = values[4] - values[3]
    temporal[f"{family}__delta34"] = delta34
    temporal[f"{family}__abs_delta34"] = delta34.abs()
    temporal[f"{family}__baseline_shift"] = values[4] - prior_median
    temporal[f"{family}__max_adjacent_change"] = adjacent.max(axis=1)
    temporal[f"{family}__temporal_range"] = (values.max(axis=1) - values.min(axis=1)).where(at_least_two)
    temporal[f"{family}__temporal_std"] = values.std(axis=1, ddof=0).where(at_least_two)

frame = pd.concat([clean[behavior], temporal], axis=1)
assert frame.shape == (25_889, 206)
values = frame.to_numpy(dtype=float)
assert np.isfinite(values[~np.isnan(values)]).all()


def empirical_tail(reference, query, top_k=8):
    ordered = np.sort(reference, axis=0)
    n_ref = len(ordered)
    surprises = np.empty(query.shape, dtype=float)
    for column in range(query.shape[1]):
        low = np.searchsorted(ordered[:, column], query[:, column], side="right")
        high = n_ref - np.searchsorted(ordered[:, column], query[:, column], side="left")
        surprises[:, column] = -np.log((np.minimum(low, high) + 1) / (n_ref + 1))
    return np.partition(surprises, -top_k, axis=1)[:, -top_k:].sum(axis=1)


def observed_tail(reference, query, top_k=8):
    surprises = np.zeros(query.shape, dtype=float)
    for column in range(query.shape[1]):
        observed_reference = np.sort(reference[:, column][np.isfinite(reference[:, column])])
        n_ref = len(observed_reference)
        assert n_ref > 2, f"Too few observed values for {frame.columns[column]}"
        observed_query = np.isfinite(query[:, column])
        query_values = query[observed_query, column]
        low = np.searchsorted(observed_reference, query_values, side="right")
        high = n_ref - np.searchsorted(observed_reference, query_values, side="left")
        surprises[observed_query, column] = -np.log((np.minimum(low, high) + 1) / (n_ref + 1))
    return np.partition(surprises, -top_k, axis=1)[:, -top_k:].sum(axis=1)


# Reproduce the scored 5% reference-trim candidate before changing missing handling.
filled = frame.fillna(frame.median()).to_numpy(dtype=float)
scaled = RobustScaler().fit_transform(filled)
old_base = empirical_tail(scaled, scaled)
retained_count = len(values) - int(np.ceil(len(values) * 0.05))
old_reference_rows = np.argsort(old_base, kind="stable")[:retained_count]
old_trimmed = empirical_tail(scaled[old_reference_rows], scaled)
old_top = set(np.argsort(-old_trimmed, kind="stable")[:2046])
old_path = ROOT / "src/submission/kaggle_t1234_peer_t5_tail_reference_trim5pct_20261121.csv"
old_submission = pd.read_csv(old_path)
assert old_top == set(np.flatnonzero(old_submission.task5.to_numpy() == 1))
print("Scored 5% Tail baseline reproduced exactly")

# Select the same fixed 5% reference trim using the observed-only score itself.
observed_base = observed_tail(values, values)
reference_rows = np.argsort(observed_base, kind="stable")[:retained_count]
scores = observed_tail(values[reference_rows], values)
assert np.isfinite(scores).all()
ranking = np.argsort(-scores, kind="stable")
new_top = set(ranking[:2046])
common = len(old_top & new_top)
cutoff_ties = int(np.count_nonzero(scores == scores[ranking[2045]]))
print(f"Observed-only ECDF | reference: {retained_count:,} | common alerts: {common:,} | swapped each way: {2046-common:,} | cutoff ties: {cutoff_ties}")

peer_path = ROOT / "src/task5/analysis/reference/peer_density_auto_v1.csv"
sample_path = ROOT / "src/submission/sample_submission.csv"
with peer_path.open(newline="", encoding="utf-8") as file:
    peer_rows = list(csv.reader(file))
sample_ids = pd.read_csv(sample_path, usecols=["id"], dtype={"id": str}).id.tolist()
assert peer_rows[0] == ["id", "task1", "task2", "task3", "task4", "task5"]
assert len(peer_rows) == len(raw) + 1
assert [row[0] for row in peer_rows[1:]] == sample_ids == old_submission.id.astype(str).tolist()
assert sum(row[5] == "1" for row in peer_rows[1:]) == 2046

predictions = np.zeros(len(values), dtype=np.int8)
predictions[ranking[:2046]] = 1
output_rows = [peer_rows[0]] + [peer[:5] + [str(int(prediction))]
                                 for peer, prediction in zip(peer_rows[1:], predictions)]
buffer = io.StringIO(newline="")
csv.writer(buffer, lineterminator="\n").writerows(output_rows)
payload = buffer.getvalue().encode("utf-8")
name = "kaggle_t1234_peer_t5_tail_observed_ecdf_trim5pct_v1_20261121"
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
    "task5_source": "F1 temporal Empirical Tail, 206 features, top-eight two-sided observed-value ECDF surprises",
    "cleaning": "Mask invalid coordinate pairs and -1 refund/guild counts; leave all other missing as NaN; no median imputation, rounding, or clipping",
    "missing_score": "Missing query features contribute zero tail surprise; each feature ECDF excludes missing reference values",
    "reference_trim": "Remove highest observed-only Tail score 5% of rows from ECDF reference only",
    "reference_rows": retained_count,
    "scored_rows": len(values),
    "task5_alerts": int(predictions.sum()),
    "common_alerts_with_scored_trim5pct": common,
    "tasks1_to_4": "teammate reference text preserved exactly row by row",
    "caveat": "No Task 5 labels, F3, or Kaggle score; all rows remain eligible for alerts",
}
card_path = output_path.with_suffix(".json")
card_payload = (json.dumps(card, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
if card_path.exists():
    assert card_path.read_bytes() == card_payload, f"Existing card differs: {card_path}"
else:
    with card_path.open("xb") as file:
        file.write(card_payload)
print(f"Exported {output_path.relative_to(ROOT)} | SHA-256 {card['csv_sha256']}")
