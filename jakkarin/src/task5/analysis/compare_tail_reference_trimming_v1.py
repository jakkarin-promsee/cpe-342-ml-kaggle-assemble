"""Compare fixed Empirical Tail rankings after excluding high-score rows from its ECDF reference.

Run from the repository root. Every supplied account remains eligible for scoring.
Pass --export to write two controlled candidate CSVs with teammate Tasks 1-4.
This sensitivity check does not estimate Task 5 F3.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--export", action="store_true")
parser.add_argument("--diagnose", action="store_true")
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
filled = frame.fillna(frame.median())
values = RobustScaler().fit_transform(filled.to_numpy(dtype=float))


def empirical_tail(reference, query, top_k=8):
    ordered = np.sort(reference, axis=0)
    n_ref = len(ordered)
    surprises = np.empty(query.shape, dtype=float)
    for column in range(query.shape[1]):
        low = np.searchsorted(ordered[:, column], query[:, column], side="right")
        high = n_ref - np.searchsorted(ordered[:, column], query[:, column], side="left")
        surprises[:, column] = -np.log((np.minimum(low, high) + 1) / (n_ref + 1))
    return np.partition(surprises, -top_k, axis=1)[:, -top_k:].sum(axis=1)


def selected(scores, budget=2046):
    return set(np.argsort(-scores, kind="stable")[:budget])


baseline = empirical_tail(values, values)
base_selected = selected(baseline)
saved = pd.read_csv(ROOT / "src/submission/kaggle_t1234_peer_t5_tail_f1_20261121.csv", dtype={"id": str})
sample_ids = pd.read_csv(ROOT / "src/submission/sample_submission.csv", usecols=["id"], dtype={"id": str})
assert len(raw) == len(saved) and saved.id.tolist() == sample_ids.id.tolist()
assert base_selected == set(np.flatnonzero(saved.task5.to_numpy() == 1))
print("Baseline reproduced exactly: 206 features, 2,046 alerts")

rows = []
if args.export:
    peer_path = ROOT / "src/task5/analysis/reference/peer_density_auto_v1.csv"
    with peer_path.open(newline="", encoding="utf-8") as file:
        peer_rows = list(csv.reader(file))
    assert peer_rows[0] == ["id", "task1", "task2", "task3", "task4", "task5"]
    assert len(peer_rows) == len(raw) + 1
    assert [row[0] for row in peer_rows[1:]] == sample_ids.id.tolist()
    assert sum(row[5] == "1" for row in peer_rows[1:]) == 2046
for trim_fraction in (0.01, 0.05):
    retained_count = len(values) - int(np.ceil(len(values) * trim_fraction))
    reference_rows = np.argsort(baseline, kind="stable")[:retained_count]
    scores = empirical_tail(values[reference_rows], values)
    current = selected(scores)
    common = len(base_selected & current)
    rows.append({
        "reference_trim_pct": 100 * trim_fraction,
        "reference_rows": retained_count,
        "common_alerts": common,
        "swapped_each_way": 2046 - common,
        "jaccard": common / (4092 - common),
        "spearman": pd.Series(baseline).corr(pd.Series(scores), method="spearman"),
        "trimmed_accounts_still_alerted": len(current - set(reference_rows)),
        "cutoff_ties": int(np.count_nonzero(scores == np.sort(scores)[-2046])),
    })
    if args.export:
        tag = f"trim{int(trim_fraction * 100)}pct"
        name = f"kaggle_t1234_peer_t5_tail_reference_{tag}_20261121"
        predictions = np.zeros(len(values), dtype=np.int8)
        predictions[np.argsort(-scores, kind="stable")[:2046]] = 1
        output_rows = [peer_rows[0]] + [peer[:5] + [str(int(prediction))]
                                         for peer, prediction in zip(peer_rows[1:], predictions)]
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerows(output_rows)
        payload = buffer.getvalue().encode("utf-8")
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
            "task5_source": "F1 temporal Empirical Tail, 206 features, top-eight two-sided ECDF surprises",
            "reference_trim": f"Remove highest original Tail score {100 * trim_fraction:g}% of rows from ECDF reference only",
            "scored_rows": len(values),
            "reference_rows": retained_count,
            "task5_alerts": int(predictions.sum()),
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
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
if args.diagnose:
    median = float(np.median(baseline))
    robust_sigma = float(1.4826 * np.median(np.abs(baseline - median)))
    threshold = median + 3 * robust_sigma
    print(f"Baseline score | median {median:.3f} | robust_sigma {robust_sigma:.3f} | median+3MAD threshold {threshold:.3f} | above {np.mean(baseline > threshold):.2%}")
    diagnostic_rows = []
    previous_top = base_selected
    for share in (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10):
        keep = len(values) - int(np.ceil(len(values) * share))
        idx = np.argsort(baseline, kind="stable")[:keep]
        current = selected(empirical_tail(values[idx], values))
        diagnostic_rows.append({"trim_pct": 100 * share,
                                "score_threshold": float(np.min(baseline[np.setdiff1d(np.arange(len(values)), idx)])),
                                "changed_from_previous": 2046 - len(current & previous_top),
                                "changed_from_zero_trim": 2046 - len(current & base_selected)})
        previous_top = current
    print(pd.DataFrame(diagnostic_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
