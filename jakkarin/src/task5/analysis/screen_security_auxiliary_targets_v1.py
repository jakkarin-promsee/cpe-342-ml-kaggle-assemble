"""Read-only screen of supplied security flags as auxiliary learning targets.

Run from the repository root. These are feature-prediction metrics, not Task 5 F3.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED=42
raw=pd.read_csv('src/task5/data/raw/test.csv')
clean=raw.copy()
for p in range(1,5):
    lat,lon=f'login_lat_{p}',f'login_lon_{p}'
    bad=(raw[lat].notna()&~raw[lat].between(-90,90))|(raw[lon].notna()&~raw[lon].between(-180,180))
    clean.loc[bad,[lat,lon]]=np.nan
for c in ['refund_count_2','refund_count_3','guild_changes_2','guild_changes_3']:
    clean.loc[raw[c].eq(-1),c]=np.nan
flags=['password_changed','vpn_usage','suspicious_login_time','mass_item_sale']
predictors=[c for c in raw if c not in ['player_id',*flags]]
fit_idx,val_idx=train_test_split(np.arange(len(clean)),test_size=.2,random_state=SEED)
X_fit=clean.iloc[fit_idx][predictors]
X_val=clean.iloc[val_idx][predictors]
for target in flags+['any_security_flag']:
    if target=='any_security_flag':
        observed=clean[flags].notna().all(axis=1)
        y=clean[flags].eq(1).any(axis=1).astype(int)
    else:
        observed=clean[target].notna()
        y=clean[target].eq(1).astype(int)
    fit_rows=observed.iloc[fit_idx].to_numpy()
    val_rows=observed.iloc[val_idx].to_numpy()
    model=LGBMClassifier(n_estimators=300,learning_rate=.04,num_leaves=15,
        min_child_samples=80,colsample_bytree=.8,subsample=.8,subsample_freq=1,
        class_weight='balanced',random_state=SEED,verbosity=-1,n_jobs=4)
    model.fit(X_fit.iloc[fit_rows],y.iloc[fit_idx].iloc[fit_rows])
    p=model.predict_proba(X_val.iloc[val_rows])[:,1]
    truth=y.iloc[val_idx].iloc[val_rows]
    print(target,'train',int(fit_rows.sum()),'val',int(val_rows.sum()),
          'prevalence',round(float(truth.mean()),4),
          'AUC',round(roc_auc_score(truth,p),4),
          'AP',round(average_precision_score(truth,p),4))
