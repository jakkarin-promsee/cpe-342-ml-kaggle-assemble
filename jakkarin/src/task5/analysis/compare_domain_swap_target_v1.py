"""Read-only probe of real versus cohort-matched, domain-swapped accounts.

Run from the repository root. This target was rejected after weak validation AUC.
"""

from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
raw = pd.read_csv('src/task5/data/raw/test.csv')
features = [c for c in raw if c != 'player_id']
clean = raw.copy()
for p in range(1, 5):
    lat, lon = f'login_lat_{p}', f'login_lon_{p}'
    invalid = (raw[lat].notna() & ~raw[lat].between(-90, 90)) | (raw[lon].notna() & ~raw[lon].between(-180, 180))
    clean.loc[invalid, [lat, lon]] = np.nan
for c in ['refund_count_2','refund_count_3','guild_changes_2','guild_changes_3']:
    clean.loc[raw[c].eq(-1), c] = np.nan

domain_families = {
    'access': ['login_count', 'login_time_variance', 'location_changes', 'device_count',
               'ip_hash_entropy', 'login_lat', 'login_lon', 'session_duration_mean', 'session_duration_std'],
    'commerce': ['currency_transfers', 'purchase_frequency', 'purchase_amount_mean',
                 'purchase_amount_std', 'refund_count'],
    'performance_social': ['skill_rating', 'skill_trajectory', 'kda_mean', 'kda_std',
                           'accuracy_mean', 'accuracy_std', 'reaction_time_mean', 'reaction_time_std',
                           'friend_adds', 'friend_removes', 'guild_changes', 'communication_frequency',
                           'activity_level', 'performance_stability', 'play_pattern_consistency'],
}
domain_columns = {name: [f'{family}_{p}' for family in families for p in range(1, 5)]
                  for name, families in domain_families.items()}
assert set(sum(domain_columns.values(), [])) == {c for c in features if c.rsplit('_',1)[-1] in {'1','2','3','4'}}
fit_idx, val_idx = train_test_split(np.arange(len(clean)), test_size=.2, random_state=SEED)
fit = clean.iloc[fit_idx][features].reset_index(drop=True)
val = clean.iloc[val_idx][features].reset_index(drop=True)


def cohort_values(frame, age_edges=None, login_edges=None):
    age = frame.account_age_days.to_numpy(dtype=float)
    login = frame[[f'login_count_{p}' for p in (1,2,3)]].median(axis=1).to_numpy(dtype=float)
    if age_edges is None:
        age_edges = np.nanquantile(age, [.25,.5,.75])
        login_edges = np.nanquantile(login, [.25,.5,.75])
    age = np.where(np.isfinite(age), age, np.nanmedian(age))
    login = np.where(np.isfinite(login), login, np.nanmedian(login))
    cohort = np.digitize(age,age_edges)*4 + np.digitize(login,login_edges)
    return cohort, age_edges, login_edges


def corrupt_domain(frame, cohort, columns, seed):
    rng = np.random.default_rng(seed)
    fake = frame.copy()
    target = frame.columns.get_indexer(columns)
    assert np.all(target >= 0)
    for group in np.unique(cohort):
        indices = np.flatnonzero(cohort == group)
        if len(indices) < 2:
            continue
        shuffled = rng.permutation(indices)
        donors = np.roll(shuffled, 1)
        fake.iloc[shuffled, target] = frame.iloc[donors, target].to_numpy()
    return fake


fit_cohort, age_edges, login_edges = cohort_values(fit)
val_cohort, _, _ = cohort_values(val, age_edges, login_edges)
fit_fake = {name: corrupt_domain(fit, fit_cohort, cols, SEED + i + 1)
            for i, (name, cols) in enumerate(domain_columns.items())}
val_fake = {name: corrupt_domain(val, val_cohort, cols, SEED + i + 101)
            for i, (name, cols) in enumerate(domain_columns.items())}
X_train = pd.concat([fit, *fit_fake.values()], ignore_index=True)
y_train = np.r_[np.zeros(len(fit),dtype=int), np.ones(len(fit)*3,dtype=int)]
model = LGBMClassifier(n_estimators=300, learning_rate=.04, num_leaves=15,
                       min_child_samples=80, colsample_bytree=.8, subsample=.8,
                       subsample_freq=1, class_weight='balanced', random_state=SEED,
                       verbosity=-1, n_jobs=4)
started = perf_counter()
model.fit(X_train,y_train)
print('fit seconds',round(perf_counter()-started,1),'rows',len(X_train))
real_score=model.predict_proba(val)[:,1]
for name,fake in val_fake.items():
    fake_score=model.predict_proba(fake)[:,1]
    auc=roc_auc_score(np.r_[np.zeros(len(val)),np.ones(len(val))],np.r_[real_score,fake_score])
    print(name,'synthetic validation AUC',round(auc,4),'real median',round(np.median(real_score),4),'fake median',round(np.median(fake_score),4))
selected=set(np.argsort(-real_score,kind='stable')[:414])
print('real score p99',np.quantile(real_score,.99),'top414 max',np.max(real_score))
importance=pd.Series(model.feature_importances_,index=features).sort_values(ascending=False)
print(importance.head(20).to_string())
