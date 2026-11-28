"""Unsupervised density and reconstruction scorer for Task 5."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer


class DensityAutoDetector(BaseEstimator):
    """Average fitted percentile ranks of GMM rarity and autoencoder error."""

    def __init__(self, anomaly_fraction=0.08, random_state=42, max_iter=40):
        self.anomaly_fraction = anomaly_fraction
        self.random_state = random_state
        self.max_iter = max_iter

    def _transform(self, X):
        if list(X.columns) != self.columns_:
            raise ValueError("Feature columns or their order differ from the fitted data")
        values = self.imputer_.transform(X)
        return np.clip(self.scaler_.transform(values), -4.0, 4.0)

    def fit(self, X, y=None):
        if not 0 < self.anomaly_fraction < 0.5:
            raise ValueError("anomaly_fraction must be between 0 and 0.5")
        self.columns_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        values = self.imputer_.transform(X)
        self.scaler_ = QuantileTransformer(
            n_quantiles=min(512, len(values)), output_distribution="normal",
            random_state=self.random_state,
        ).fit(values)
        values = self._transform(X)
        self.pca_ = PCA(n_components=min(12, values.shape[1], len(values) - 1),
                        random_state=self.random_state).fit(values)
        latent = self.pca_.transform(values)
        self.density_ = GaussianMixture(
            n_components=4, covariance_type="diag", reg_covar=1e-4,
            max_iter=120, random_state=self.random_state,
        ).fit(latent)
        self.autoencoder_ = MLPRegressor(
            hidden_layer_sizes=(64, 16, 64), alpha=1e-3, batch_size=256,
            max_iter=self.max_iter, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=5, random_state=self.random_state,
        ).fit(values, values)
        density, reconstruction = self.component_scores(X)
        self.density_reference_ = np.sort(density)
        self.reconstruction_reference_ = np.sort(reconstruction)
        scores = self.anomaly_score(X)
        self.threshold_ = float(np.quantile(scores, 1 - self.anomaly_fraction))
        return self

    def component_scores(self, X):
        values = self._transform(X)
        density = -self.density_.score_samples(self.pca_.transform(values))
        reconstruction = np.mean((values - self.autoencoder_.predict(values)) ** 2, axis=1)
        return density, reconstruction

    def anomaly_score(self, X):
        density, reconstruction = self.component_scores(X)
        n = len(self.density_reference_)
        density_rank = np.searchsorted(self.density_reference_, density, side="right") / n
        reconstruction_rank = np.searchsorted(self.reconstruction_reference_, reconstruction, side="right") / n
        return 0.5 * (density_rank + reconstruction_rank)

    def predict(self, X):
        return (self.anomaly_score(X) >= self.threshold_).astype("int8")
