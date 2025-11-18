#!/usr/bin/env python3
"""
Phase 3 (Unified): Aging Clock Model Building for 8 Models
- Minimal console output, tqdm progress bars
- No plotting; only nested CV + final model training
- Uses same CV settings and hyperparameter grids as individual scripts
- 1-SE principle removed where present; use best CV parameters directly
- Outputs a single CSV: result/3_unified_model_summary.csv

Date: 2025-11-18T08:14:32.428Z
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm

# Common sklearn tools
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Linear models
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
# PLSR
from sklearn.cross_decomposition import PLSRegression
# SVR
from sklearn.svm import SVR as SKSVR
# Tree ensembles
from sklearn.ensemble import RandomForestRegressor
# Gradient boosting
import xgboost as xgb
import lightgbm as lgb


def load_data(result_dir="result"):
    print("Loading data...")
    # Phase 2 selected features
    selected_features_path = os.path.join(result_dir, "2_selected_features.csv")
    selected_features = pd.read_csv(selected_features_path)
    cpg_sites = selected_features['CpG_site'].tolist()

    # Phase 1 preprocessed data
    data_path = os.path.join(result_dir, "1_preprocessed_data.csv")
    data = pd.read_csv(data_path, index_col=0)

    X = data[cpg_sites].copy()
    y = data['Age'].values

    if X.isnull().values.any():
        X = X.fillna(X.median())

    return X.values, y, cpg_sites, data.index.values


def _metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    corr = np.corrcoef(actual, pred)[0, 1]
    return mae, rmse, r2, corr


def run_lasso(X, y):
    n = len(y)
    loo = LeaveOneOut()
    preds = np.zeros(n)
    scaler = StandardScaler()
    alphas = np.logspace(-4, 1, 50)
    for i, (tr, te) in enumerate(tqdm(loo.split(X), total=n, desc="LASSO LOOCV", leave=False)):
        Xtr, Xte = X[tr], X[te]; ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
        lcv = LassoCV(alphas=alphas, cv=LeaveOneOut(),
                      max_iter=50000, tol=1e-3, n_jobs=-1, random_state=42)
        lcv.fit(Xtr_s, ytr)
        # Use best alpha directly (no 1-SE)
        final = LassoCV(alphas=[lcv.alpha_], cv=LeaveOneOut(),
                        max_iter=50000, tol=1e-3, n_jobs=-1, random_state=42)
        final.fit(Xtr_s, ytr)
        preds[te[0]] = final.predict(Xte_s)[0]
    mae, rmse, r2, corr = _metrics(y, preds)
    params = {"alpha": float(lcv.alpha_) if 'lcv' in locals() else None}
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "Correlation": corr, "params": params}


def run_elastic_net(X, y):
    n = len(y)
    loo = LeaveOneOut()
    preds = np.zeros(n)
    scaler = StandardScaler()
    alphas = np.logspace(-4, 1, 50)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i, (tr, te) in enumerate(tqdm(loo.split(X), total=n, desc="ElasticNet LOOCV", leave=False)):
        Xtr, Xte = X[tr], X[te]; ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
        encv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=LeaveOneOut(),
                            max_iter=50000, tol=1e-3, n_jobs=-1, random_state=42, selection='cyclic')
        encv.fit(Xtr_s, ytr)
        final = ElasticNetCV(alphas=[encv.alpha_], l1_ratio=[float(encv.l1_ratio_)],
                             cv=KFold(n_splits=5, shuffle=True, random_state=42), max_iter=50000,
                             tol=1e-3, n_jobs=-1, random_state=42, selection='cyclic')
        final.fit(Xtr_s, ytr)
        preds[te[0]] = final.predict(Xte_s)[0]
    mae, rmse, r2, corr = _metrics(y, preds)
    params = {"alpha": float(encv.alpha_), "l1_ratio": float(encv.l1_ratio_)} if 'encv' in locals() else {}
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "Correlation": corr, "params": params}


def run_ridge(X, y):
    n = len(y)
    loo = LeaveOneOut()
    preds = np.zeros(n)
    scaler = StandardScaler()
    alphas = np.logspace(-4, 4, 100)
    inner = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (tr, te) in enumerate(tqdm(loo.split(X), total=n, desc="Ridge LOOCV", leave=False)):
        Xtr, Xte = X[tr], X[te]; ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
        rcv = RidgeCV(alphas=alphas, cv=LeaveOneOut(), scoring='neg_mean_absolute_error')
        rcv.fit(Xtr_s, ytr)
        preds[te[0]] = rcv.predict(Xte_s)[0]
    mae, rmse, r2, corr = _metrics(y, preds)
    params = {"alpha": float(rcv.alpha_)} if 'rcv' in locals() else {}
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "Correlation": corr, "params": params}


def run_plsr(X, y):
    n = len(y)
    loo = LeaveOneOut()
    preds = np.zeros(n)
    scaler = StandardScaler()
    for i, (tr, te) in enumerate(tqdm(loo.split(X), total=n, desc="PLSR LOOCV", leave=False)):
        Xtr, Xte = X[tr], X[te]; ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
        # Inner CV to select components (1..20 or limited by data)
        max_components = min(Xtr_s.shape[1], len(ytr) - 1)
        component_range = range(1, min(max_components + 1, 21))
        inner = KFold(n_splits=5, shuffle=True, random_state=42)
        best_comp, best_score = 1, float('inf')
        for ncomp in component_range:
            scores = []
            for itr, ival in inner.split(Xtr_s):
                pls = PLSRegression(n_components=ncomp)
                pls.fit(Xtr_s[itr], ytr[itr])
                pred = pls.predict(Xtr_s[ival])
                scores.append(mean_absolute_error(ytr[ival], pred))
            avg = np.mean(scores)
            if avg < best_score:
                best_score, best_comp = avg, ncomp
        final = PLSRegression(n_components=best_comp)
        final.fit(Xtr_s, ytr)
        preds[te[0]] = final.predict(Xte_s)[0]
    mae, rmse, r2, corr = _metrics(y, preds)
    params = {"n_components": int(best_comp)} if 'best_comp' in locals() else {}
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "Correlation": corr, "params": params}


def run_svr(X, y):
    n = len(y)
    loo = LeaveOneOut()
    preds = np.zeros(n)
    scaler = StandardScaler()
    param_grids = {
        'linear': {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 0.5, 1.0]},
        'rbf': {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 0.5], 'gamma': ['scale', 0.001, 0.01, 0.1]},
    }
    for i, (tr, te) in enumerate(tqdm(loo.split(X), total=n, desc="SVR LOOCV", leave=False)):
        Xtr, Xte = X[tr], X[te]; ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
        inner = KFold(n_splits=5, shuffle=True, random_state=42)
        best_params, best_score = None, float('inf')
        for kernel, grid in param_grids.items():
            keys = list(grid.keys())
            from itertools import product
            for values in product(*[grid[k] for k in keys]):
                cfg = dict(zip(keys, values)); cfg['kernel'] = kernel
                scores = []
                for itr, ival in inner.split(Xtr_s):
                    svr = SKSVR(**cfg)
                    svr.fit(Xtr_s[itr], ytr[itr])
                    pred = svr.predict(Xtr_s[ival])
                    scores.append(mean_absolute_error(ytr[ival], pred))
                avg = np.mean(scores)
                if avg < best_score:
                    best_score, best_params = avg, cfg.copy()
        final = SKSVR(**best_params)
        final.fit(Xtr_s, ytr)
        preds[te[0]] = final.predict(Xte_s)[0]
    mae, rmse, r2, corr = _metrics(y, preds)
    params = best_params if 'best_params' in locals() else {}
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "Correlation": corr, "params": params}


def run_rf(X, y):
    n = len(y)
    loo = LeaveOneOut()
    preds = np.zeros(n)
    param_grid = [
        {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 7, 'max_features': 'sqrt'},
        {'n_estimators': 300, 'max_depth': 5, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 0.5},
        {'n_estimators': 300, 'max_depth': 7, 'min_samples_split': 20, 'min_samples_leaf': 7, 'max_features': 0.5},
        {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 500, 'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 7, 'max_features': 0.5},
        {'n_estimators': 500, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 700, 'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 0.7},
    ]
    for i, (tr, te) in enumerate(tqdm(loo.split(X), total=n, desc="RF LOOCV", leave=False)):
        Xtr, Xte = X[tr], X[te]; ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        inner = KFold(n_splits=5, shuffle=True, random_state=42)
        best_params, best_score, best_oob = None, float('inf'), -np.inf
        for params in tqdm(param_grid, desc=f"RF inner (fold {i+1})", leave=False):
            scores, oobs = [], []
            for itr, ival in inner.split(Xtr):
                rf = RandomForestRegressor(**params, oob_score=True, random_state=42, n_jobs=2)
                rf.fit(Xtr[itr], ytr[itr])
                pred = rf.predict(Xtr[ival])
                scores.append(mean_absolute_error(ytr[ival], pred))
                if hasattr(rf, 'oob_score_'):
                    oobs.append(rf.oob_score_)
            avg, avg_oob = np.mean(scores), (np.mean(oobs) if oobs else 0)
            if avg < best_score or (avg == best_score and avg_oob > best_oob):
                best_score, best_oob = avg, avg_oob
                best_params = params.copy()
        final = RandomForestRegressor(**best_params, oob_score=True, random_state=42, n_jobs=-1)
        final.fit(Xtr, ytr)
        preds[te[0]] = final.predict(Xte)[0]
    mae, rmse, r2, corr = _metrics(y, preds)
    params = best_params if 'best_params' in locals() else {}
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "Correlation": corr, "params": params}


def run_xgb(X, y):
    n = len(y)
    loo = LeaveOneOut()
    preds = np.zeros(n)
    param_grid = [
        {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.1, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 10, 'subsample': 0.6, 'colsample_bytree': 0.6},
        {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.1, 'reg_alpha': 3.0, 'reg_lambda': 7.0, 'min_child_weight': 15, 'subsample': 0.5, 'colsample_bytree': 0.5},
        {'n_estimators': 300, 'max_depth': 2, 'learning_rate': 0.05, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 20, 'subsample': 0.4, 'colsample_bytree': 0.4},
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 8, 'subsample': 0.7, 'colsample_bytree': 0.7},
        {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 12, 'subsample': 0.6, 'colsample_bytree': 0.6},
        {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05, 'reg_alpha': 3.0, 'reg_lambda': 7.0, 'min_child_weight': 15, 'subsample': 0.5, 'colsample_bytree': 0.5},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.1, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 8, 'subsample': 0.7, 'colsample_bytree': 0.7},
        {'n_estimators': 700, 'max_depth': 4, 'learning_rate': 0.03, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 10, 'subsample': 0.6, 'colsample_bytree': 0.6},
        {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.02, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 8, 'subsample': 0.7, 'colsample_bytree': 0.7},
        {'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.01, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 12, 'subsample': 0.6, 'colsample_bytree': 0.6},
        {'n_estimators': 1500, 'max_depth': 3, 'learning_rate': 0.005, 'reg_alpha': 3.0, 'reg_lambda': 7.0, 'min_child_weight': 15, 'subsample': 0.5, 'colsample_bytree': 0.5},
        {'n_estimators': 1000, 'max_depth': 2, 'learning_rate': 0.01, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 20, 'subsample': 0.4, 'colsample_bytree': 0.4},
        {'n_estimators': 1500, 'max_depth': 2, 'learning_rate': 0.005, 'reg_alpha': 7.0, 'reg_lambda': 15.0, 'min_child_weight': 25, 'subsample': 0.3, 'colsample_bytree': 0.3},
    ]
    for i, (tr, te) in enumerate(tqdm(loo.split(X), total=n, desc="XGBoost LOOCV", leave=False)):
        Xtr, Xte = X[tr], X[te]; ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        inner = KFold(n_splits=5, shuffle=True, random_state=42)
        best_params, best_score = None, float('inf')
        for params in tqdm(param_grid, desc=f"XGB inner (fold {i+1})", leave=False):
            scores = []
            for itr, ival in inner.split(Xtr):
                mdl = xgb.XGBRegressor(**params, random_state=42, n_jobs=1, verbosity=0)
                mdl.fit(Xtr[itr], ytr[itr], verbose=False)
                pred = mdl.predict(Xtr[ival])
                scores.append(mean_absolute_error(ytr[ival], pred))
            avg = np.mean(scores)
            if avg < best_score:
                best_score, best_params = avg, params.copy()
        final = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        final.fit(Xtr, ytr, verbose=False)
        preds[te[0]] = final.predict(Xte)[0]
    mae, rmse, r2, corr = _metrics(y, preds)
    params = best_params if 'best_params' in locals() else {}
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "Correlation": corr, "params": params}


def run_lgbm(X, y):
    n = len(y)
    loo = LeaveOneOut()
    preds = np.zeros(n)
    param_grid = [
        {'n_estimators': 100, 'num_leaves': 10, 'learning_rate': 0.1, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 15, 'lambda_l1': 1.0, 'lambda_l2': 2.0},
        {'n_estimators': 200, 'num_leaves': 15, 'learning_rate': 0.1, 'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'min_data_in_leaf': 20, 'lambda_l1': 2.0, 'lambda_l2': 3.0},
        {'n_estimators': 300, 'num_leaves': 20, 'learning_rate': 0.05, 'feature_fraction': 0.4, 'bagging_fraction': 0.4, 'min_data_in_leaf': 25, 'lambda_l1': 3.0, 'lambda_l2': 5.0},
        {'n_estimators': 200, 'num_leaves': 25, 'learning_rate': 0.1, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'min_data_in_leaf': 10, 'lambda_l1': 0.5, 'lambda_l2': 2.0},
        {'n_estimators': 300, 'num_leaves': 30, 'learning_rate': 0.1, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 15, 'lambda_l1': 1.0, 'lambda_l2': 3.0},
        {'n_estimators': 500, 'num_leaves': 35, 'learning_rate': 0.05, 'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'min_data_in_leaf': 20, 'lambda_l1': 2.0, 'lambda_l2': 5.0},
        {'n_estimators': 300, 'num_leaves': 40, 'learning_rate': 0.1, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'min_data_in_leaf': 8, 'lambda_l1': 0.2, 'lambda_l2': 1.0},
        {'n_estimators': 500, 'num_leaves': 45, 'learning_rate': 0.05, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'min_data_in_leaf': 10, 'lambda_l1': 0.5, 'lambda_l2': 2.0},
        {'n_estimators': 700, 'num_leaves': 50, 'learning_rate': 0.03, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 12, 'lambda_l1': 1.0, 'lambda_l2': 3.0},
        {'n_estimators': 500, 'num_leaves': 25, 'learning_rate': 0.02, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'min_data_in_leaf': 10, 'lambda_l1': 0.5, 'lambda_l2': 2.0},
        {'n_estimators': 1000, 'num_leaves': 30, 'learning_rate': 0.01, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'min_data_in_leaf': 12, 'lambda_l1': 1.0, 'lambda_l2': 3.0},
        {'n_estimators': 1500, 'num_leaves': 35, 'learning_rate': 0.005, 'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'min_data_in_leaf': 15, 'lambda_l1': 2.0, 'lambda_l2': 5.0},
        {'n_estimators': 1000, 'num_leaves': 20, 'learning_rate': 0.01, 'feature_fraction': 0.4, 'bagging_fraction': 0.4, 'min_data_in_leaf': 20, 'lambda_l1': 3.0, 'lambda_l2': 7.0},
        {'n_estimators': 1500, 'num_leaves': 25, 'learning_rate': 0.005, 'feature_fraction': 0.3, 'bagging_fraction': 0.3, 'min_data_in_leaf': 25, 'lambda_l1': 5.0, 'lambda_l2': 10.0},
    ]
    for i, (tr, te) in enumerate(tqdm(loo.split(X), total=n, desc="LightGBM LOOCV", leave=False)):
        Xtr, Xte = X[tr], X[te]; ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        inner = KFold(n_splits=5, shuffle=True, random_state=42)
        best_params, best_score = None, float('inf')
        for params in tqdm(param_grid, desc=f"LGBM inner (fold {i+1})", leave=False):
            scores = []
            for itr, ival in inner.split(Xtr):
                mdl = lgb.LGBMRegressor(**params, random_state=42, n_jobs=1, verbosity=-1)
                mdl.fit(Xtr[itr], ytr[itr])
                pred = mdl.predict(Xtr[ival])
                scores.append(mean_absolute_error(ytr[ival], pred))
            avg = np.mean(scores)
            if avg < best_score:
                best_score, best_params = avg, params.copy()
        final = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
        final.fit(Xtr, ytr)
        preds[te[0]] = final.predict(Xte)[0]
    mae, rmse, r2, corr = _metrics(y, preds)
    params = best_params if 'best_params' in locals() else {}
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "Correlation": corr, "params": params}


def main():
    X, y, cpg_sites, sample_ids = load_data()
    os.makedirs('result', exist_ok=True)

    models = [
        ("LASSO", run_lasso),
        ("Elastic Net", run_elastic_net),
        ("Ridge", run_ridge),
        ("PLSR", run_plsr),
        ("SVR", run_svr),
        ("Random Forest", run_rf),
        ("XGBoost", run_xgb),
        ("LightGBM", run_lgbm),
    ]

    rows = []
    for name, fn in tqdm(models, desc="Models", leave=True):
        preds, res = fn(X, y)
        rows.append({
            "model": name,
            "MAE": res["MAE"],
            "RMSE": res["RMSE"],
            "R2": res["R2"],
            "Correlation": res["Correlation"],
            "n_samples": len(y),
            "n_features": X.shape[1],
            "best_params": json.dumps(res.get("params", {})),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv('result/3_unified_model_summary.csv', index=False)
    print("Saved: result/3_unified_model_summary.csv")


if __name__ == "__main__":
    main()
