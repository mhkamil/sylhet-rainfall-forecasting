"""Stage 4 model factories: factory(seed, y_train) -> unfitted estimator. Default hyper-parameters, not tuned.

Linear models get a Pipeline (one-hot month + standardise everything else) so preprocessing is fitted on training
rows only. Tree models use raw features (splits are invariant to monotone rescaling).
scikit-learn 1.9: LogisticRegression's `penalty` is deprecated -> no penalty = C=inf, L2 = l1_ratio=0,
L1 = l1_ratio=1 (saga), elastic net = 0 < l1_ratio < 1 (saga).
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, SGDClassifier,
                                  SGDRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEEDS = [0, 1, 2, 3, 4]


def linear_pre():
    return ColumnTransformer([("month", OneHotEncoder(drop="first", sparse_output=False), ["month"])],
                             remainder=StandardScaler())


def _lin(model):
    return lambda seed, y: make_pipeline(linear_pre(), model(seed))


# name: (factory, stochastic?)  -- stochastic models are run with 5 seeds, deterministic ones with seed 0
LINEAR_CLS = {
    "logreg_none": (_lin(lambda s: LogisticRegression(C=np.inf, class_weight="balanced", max_iter=5000)), False),
    "logreg_l2": (_lin(lambda s: LogisticRegression(C=1.0, l1_ratio=0.0, class_weight="balanced", max_iter=5000)), False),
    "logreg_l1": (_lin(lambda s: LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", class_weight="balanced",
                                                    max_iter=5000, random_state=s)), True),
    "logreg_enet": (_lin(lambda s: LogisticRegression(C=0.1, l1_ratio=0.5, solver="saga", class_weight="balanced",
                                                      max_iter=5000, random_state=s)), True),
    "sgd_logloss": (_lin(lambda s: SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced",
                                                 random_state=s)), True),
}
LINEAR_REG = {
    "linreg": (_lin(lambda s: LinearRegression()), False),
    "ridge": (_lin(lambda s: Ridge(alpha=1.0)), False),
    "lasso": (_lin(lambda s: Lasso(alpha=0.1, max_iter=20000)), False),
    "elasticnet": (_lin(lambda s: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000)), False),
    "sgd_reg": (_lin(lambda s: SGDRegressor(alpha=1e-4, random_state=s)), True),
}


def _pos_weight(y):
    y = np.asarray(y)
    return (y == 0).sum() / max((y == 1).sum(), 1)


def _xgb(task):
    import xgboost as xgb
    common = dict(n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=5, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4)
    if task == "cls":
        return lambda s, y: xgb.XGBClassifier(**common, eval_metric="logloss", scale_pos_weight=_pos_weight(y),
                                              random_state=s)
    return lambda s, y: xgb.XGBRegressor(**common, eval_metric="rmse", random_state=s)


def _lgbm(task):
    import lightgbm as lgb
    common = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbose=-1)
    if task == "cls":
        return lambda s, y: lgb.LGBMClassifier(**common, class_weight="balanced", random_state=s)
    return lambda s, y: lgb.LGBMRegressor(**common, random_state=s)


def _cat(task):
    import catboost as cb
    common = dict(iterations=2000, learning_rate=0.05, depth=6, thread_count=4, verbose=0, allow_writing_files=False)
    if task == "cls":
        return lambda s, y: cb.CatBoostClassifier(**common, auto_class_weights="Balanced", random_seed=s)
    return lambda s, y: cb.CatBoostRegressor(**common, loss_function="RMSE", random_seed=s)


def _rf(task):
    if task == "cls":
        return lambda s, y: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                   class_weight="balanced", n_jobs=4, random_state=s)
    return lambda s, y: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33,
                                              n_jobs=4, random_state=s)


def trees(task):
    """name -> factory for the tree family; all are stochastic (5 seeds)."""
    return {"random_forest": _rf(task), "xgboost": _xgb(task), "lightgbm": _lgbm(task), "catboost": _cat(task)}
