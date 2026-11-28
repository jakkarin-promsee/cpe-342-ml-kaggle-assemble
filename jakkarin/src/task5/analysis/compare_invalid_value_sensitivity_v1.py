"""Read-only Task 5 sensitivity check for raw invalid values.

Refits the fixed full-file detectors without external labels or Kaggle calls.
Compares top-2,046 selections under cleaned versus unmasked invalid values.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

raw = pd.read_csv('src/task5/data/raw/test.csv')
clean = raw.copy()
features = [c for c in raw if c != 'player_id']
security = ['password_changed', 'vpn_usage', 'suspicious_login_time', 'mass_item_sale']
coordinates = [f'login_{coord}_{period}' for period in range(1, 5) for coord in ('lat', 'lon')]
invalid_counts = ['refund_count_2', 'refund_count_3', 'guild_changes_2', 'guild_changes_3']
behavior = [c for c in features if c not in security and c not in coordinates]
for period in range(1, 5):
    lat, lon = f'login_lat_{period}', f'login_lon_{period}'
    invalid = (raw[lat].notna() & ~raw[lat].between(-90, 90)) | (raw[lon].notna() & ~raw[lon].between(-180, 180))
    clean.loc[invalid, [lat, lon]] = np.nan
for c in invalid_counts:
    clean.loc[raw[c].eq(-1), c] = np.nan

families = [
    'login_count', 'login_time_variance', 'location_changes', 'device_count',
    'ip_hash_entropy', 'activity_level', 'session_duration_mean',
    'play_pattern_consistency', 'skill_rating', 'skill_trajectory',
    'kda_mean', 'reaction_time_mean', 'currency_transfers',
    'purchase_amount_mean', 'friend_removes', 'communication_frequency',
]
temporal = pd.DataFrame(index=clean.index)
for family in families:
    values = clean[[f'{family}_{period}' for period in range(1, 5)]].copy()
    values.columns = range(1, 5)
    earlier_median = values[[1, 2, 3]].median(axis=1)
    adjacent = pd.concat([(values[period + 1] - values[period]).abs() for period in range(1, 4)], axis=1)
    has_two = values.notna().sum(axis=1).ge(2)
    delta34 = values[4] - values[3]
    temporal[f'{family}__delta34'] = delta34
    temporal[f'{family}__abs_delta34'] = delta34.abs()
    temporal[f'{family}__baseline_shift'] = values[4] - earlier_median
    temporal[f'{family}__max_adjacent_change'] = adjacent.max(axis=1)
    temporal[f'{family}__temporal_range'] = (values.max(axis=1) - values.min(axis=1)).where(has_two)
    temporal[f'{family}__temporal_std'] = values.std(axis=1, ddof=0).where(has_two)


def matrix(frame, scale=False):
    values = frame.fillna(frame.median()).to_numpy(dtype=float)
    return RobustScaler().fit_transform(values) if scale else values


def tail_scores(values, top_k=8):
    sorted_fit = np.sort(values, axis=0)
    n = len(sorted_fit)
    tails = np.empty_like(values)
    for j in range(values.shape[1]):
        left = np.searchsorted(sorted_fit[:, j], values[:, j], side='right')
        right = n - np.searchsorted(sorted_fit[:, j], values[:, j], side='left')
        tails[:, j] = -np.log((np.minimum(left, right) + 1) / (n + 1))
    return np.partition(tails, -top_k, axis=1)[:, -top_k:].sum(axis=1)


def selected(scores, count=2046):
    return set(np.argsort(-scores, kind='stable')[:count])


def compare(base, other):
    common = len(base & other)
    return common, 2046 - common, common / (4092 - common)


def csv_selected(path):
    frame = pd.read_csv(path)
    return set(np.flatnonzero(frame.task5.to_numpy() == 1))


clean_tail_values = matrix(pd.concat((clean[behavior], temporal), axis=1), scale=True)
raw_tail_values = matrix(pd.concat((raw[behavior], temporal), axis=1), scale=True)
clean_tail = selected(tail_scores(clean_tail_values))
raw_tail = selected(tail_scores(raw_tail_values))
assert clean_tail == csv_selected('src/submission/kaggle_t1234_peer_t5_tail_f1_20261121.csv')
print('TAIL_F1 clean vs keep -1:', compare(clean_tail, raw_tail))

clean_ocsvm_values = matrix(clean[behavior], scale=True)
raw_ocsvm_values = matrix(raw[behavior], scale=True)
clean_ocsvm = selected(-OneClassSVM(kernel='rbf', nu=.1, gamma='scale', cache_size=500).fit(clean_ocsvm_values).score_samples(clean_ocsvm_values))
raw_ocsvm = selected(-OneClassSVM(kernel='rbf', nu=.1, gamma='scale', cache_size=500).fit(raw_ocsvm_values).score_samples(raw_ocsvm_values))
assert clean_ocsvm == csv_selected('src/submission/kaggle_t1234_peer_t5_ocsvm_f0_20261121.csv')
print('OCSVM_F0 clean vs keep -1:', compare(clean_ocsvm, raw_ocsvm))

if_params = {'n_estimators': 300, 'max_samples': 256, 'contamination': 'auto', 'random_state': 42, 'n_jobs': -1}
variants = {'clean': clean[features].copy()}
variants['keep_invalid_counts'] = clean[features].copy()
variants['keep_invalid_counts'][invalid_counts] = raw[invalid_counts]
variants['keep_invalid_coordinates'] = clean[features].copy()
variants['keep_invalid_coordinates'][coordinates] = raw[coordinates]
variants['keep_both'] = raw[features].copy()
if_selections = {}
for name, frame in variants.items():
    values = matrix(frame)
    scores = -IsolationForest(**if_params).fit(values).score_samples(values)
    if_selections[name] = selected(scores)
for name in ('keep_invalid_counts', 'keep_invalid_coordinates', 'keep_both'):
    print('IF_F6 clean vs', name, compare(if_selections['clean'], if_selections[name]))

print('invalid count players', int(raw[invalid_counts].eq(-1).any(axis=1).sum()))
print('invalid coordinate players', int((raw[coordinates].ne(clean[coordinates]) & raw[coordinates].notna()).any(axis=1).sum()))
