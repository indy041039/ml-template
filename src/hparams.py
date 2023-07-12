from optuna.samplers import TPESampler
import optuna

from copy import deepcopy
from src.evaluate import *
import numpy as np


def xgb_params(trial):
  params = {
      "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
      "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
      "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
      "subsample": trial.suggest_float("subsample", 0.6, 1.0),
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
      "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
      "max_depth": trial.suggest_int("max_depth", 3, 18),
      "n_estimators": trial.suggest_categorical("n_estimators", [1000]),
      "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1,2,3,4,5,6,7,8,9,10]),
      "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
      "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
  }
  return params


def xgb_objective(trial, model, X, y, early_stopping_rounds, cv, scoring, random_state):
  params = xgb_params(trial)
  scores = []
  model = model.set_params(**params)

  for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=early_stopping_rounds)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]

    if scoring in [roc_auc_score, average_precision_score]:
      scores.append(scoring(y_val, y_prob))
    else:
      scores.append(scoring(y_val, y_pred))
  return np.mean(scores)


def lgb_params(trial):
  params = {
      "n_estimators": trial.suggest_categorical("n_estimators", [1000]),
      "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
      "num_leaves": trial.suggest_int("num_leaves", 2, 512),
      "max_depth": trial.suggest_int("max_depth", 3, 12),
      "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
      "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
      "subsample": trial.suggest_float("subsample", 0.6, 1.0),
      "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
      "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
      "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1,2,3,4,5,6,7,8,9,10]),
  }
  return params


def lgb_objective(trial, model, X, y, early_stopping_rounds, cv, scoring, random_state):
  params = lgb_params(trial)
  scores = []
  model = model.set_params(**params)

  for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              eval_metric='binary_logloss',
              verbose=False,
              early_stopping_rounds=early_stopping_rounds)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]

    if scoring in [roc_auc_score, average_precision_score]:
      scores.append(scoring(y_val, y_prob))
    else:
      scores.append(scoring(y_val, y_pred))

  return np.mean(scores)


def cat_params(trial):
  params = {
      "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
      "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
      "depth": trial.suggest_int("depth", 1, 12),
      "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
      "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
  }
  if params["bootstrap_type"] == "Bayesian":
      params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
  elif params["bootstrap_type"] == "Bernoulli":
      params["subsample"] = trial.suggest_float("subsample", 0.5, 1)
  return params


def cat_objective(trial, model, X, y, early_stopping_rounds, cv, scoring, random_state):
  params = cat_params(trial)
  scores = []
  # model = CatBoostClassifier(random_state=random_state, **params)
  model = deepcopy(model)
  model = model.set_params(**params)

  for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              verbose=0,
              early_stopping_rounds=early_stopping_rounds)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]

    if scoring in [roc_auc_score, average_precision_score]:
      scores.append(scoring(y_val, y_prob))
    else:
      scores.append(scoring(y_val, y_pred))

  return np.mean(scores)


def rf_params(trial):
  params = {
      'max_depth': trial.suggest_categorical('max_depth', [3,5,7,9,12,None]),
      'n_estimators': trial.suggest_categorical('n_estimators', [50,100,250]),
      'max_samples': trial.suggest_categorical('max_samples', [0.6,0.8,None]),
      'max_features': trial.suggest_categorical('max_features', [0.6,0.8,'auto',None]),
      'min_samples_split': trial.suggest_categorical('min_samples_split', [2,5,10]),
      'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1,2,4])
  }
  return params


def rf_objective(trial, model, X, y, cv, scoring, random_state):
  params = rf_params(trial)
  scores = []
  model = model.set_params(**params)
  
  for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]

    if scoring in [roc_auc_score, average_precision_score]:
      scores.append(scoring(y_val, y_prob))
    else:
      scores.append(scoring(y_val, y_pred))

  return np.mean(scores)
