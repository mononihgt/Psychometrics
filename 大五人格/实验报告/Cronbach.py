import pandas as pd
import numpy as np
from scipy.stats import f
from sklearn.preprocessing import StandardScaler


__call__ = ["CronbachAlpha"]

class CronbachAlpha:
    def __init__(self, nan_policy="pairwise", ci = 0.95):
        assert nan_policy in ["pairwise", "listwise"], "nan_policy must be 'pairwise' or 'listwise'."
        self.nan_policy = nan_policy
        self.ci = ci

    def __call__(self, data, items=None, scores=None, subject=None):
        assert isinstance(data, pd.DataFrame), "data must be a dataframe."
        if all([v is not None for v in [items, scores, subject]]):
            # Data in long-format: we first convert to a wide format
            data = data.pivot(index=subject, values=scores, columns=items)

        n, k = data.shape
        assert k >= 2, "At least two items are required."
        assert n >= 2, "At least two raters/subjects are required."
        err = "All columns must be numeric."
        assert all([data[c].dtype.kind in "bfiu" for c in data.columns]), err
        if data.isna().any().any() and self.nan_policy == "listwise":
            data = data.dropna(axis=0, how="any")

        # Compute covariance matrix and Cronbach's alpha
        # Standardize the data using sklearn
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        C = data.cov(numeric_only=True)
        cronbach = (k / (k - 1)) * (1 - np.trace(C) / C.sum().sum())

        alpha = 1 - self.ci
        df1 = n - 1
        df2 = df1 * (k - 1)
        lower = 1 - (1 - cronbach) * f.isf(alpha / 2, df1, df2)
        upper = 1 - (1 - cronbach) * f.isf(1 - alpha / 2, df1, df2)

        # calculate alpha.drop,which is the alpha value if one item is dropped
        alpha_drop = pd.Series(index=data.columns)
        for item in data.columns:
            C_drop = data.drop(item, axis=1).cov(numeric_only=True)
            alpha_drop[item] = (k / (k - 1)) * (1 - np.trace(C_drop) / C_drop.sum().sum())
        return cronbach, np.round([lower, upper], 3), alpha_drop
        
