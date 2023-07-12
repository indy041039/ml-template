from src.evaluate import *
from src.logger import *
from src.hparams import *
from src.utils import *

from copy import deepcopy
import os 
import glob
import json
import joblib


class ClassificationPipeline():
  def __init__(self, model, preprocessor, random_state,
               tune_model=False, scoring=None, cv=None, 
               n_trials=None, timeout=None, logger=None):
    
    # Train Config
    self.model = deepcopy(model) # RandomForest, XGBoost, LightGBM, CatBoost
    self.preprocessor = deepcopy(preprocessor)
    self.random_state = random_state
    
    # Hyperparameter Tuning
    self.tune_model = tune_model
    self.n_trials = n_trials
    self.scoring = scoring
    self.timeout = timeout
    self.cv = cv

    if not logger:
      self.logger = logging.getLogger()
    else:
      self.logger = logger

    self.model_name = self.model.__class__.__name__

  def train(self, X_train, y_train):
    X_train, y_train = X_train.copy(), y_train.copy()

    self.logger.info(f' Train model: {self.model_name} '.center(50, '='))
    self.logger.info(f'Input shape: {X_train.shape}')
    self.logger.info(f'Target ratio: {y_train.value_counts(normalize=True).tolist()}')

    self.in_feature = X_train.columns.tolist()

    # Preprocess
    self.logger.info('Preprocess data')
    X_train = self.preprocessor.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=self.preprocessor.get_feature_names_out())
    self.out_feature = X_train.columns.tolist()

    # Train Model
    if self.tune_model:
      self.logger.info(f'Tune model ({self.model_name})')
      self.logger.info(f'Maximize metric: {self.scoring}')
      self.logger.info(f'Number of trials: {self.n_trials}')
      self.logger.info(f'Timeout: {self.timeout} seconds')
      if self.model_name == 'XGBClassifier':
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(lambda trial: xgb_objective(trial, self.model, X_train, y_train,
                                                   scoring=self.scoring,
                                                   early_stopping_rounds=50,
                                                   cv=self.cv, random_state=self.random_state), 
                      n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
        self.best_params = study.best_params
        
      elif self.model_name == 'LGBMClassifier':
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(lambda trial: lgb_objective(trial, self.model, X_train, y_train,
                                                   scoring=self.scoring,
                                                   early_stopping_rounds=50,
                                                   cv=self.cv, random_state=self.random_state), 
                      n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
        self.best_params = study.best_params

      elif self.model_name == 'CatBoostClassifier':
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(lambda trial: cat_objective(trial, self.model, X_train, y_train,
                                                   scoring=self.scoring,
                                                   early_stopping_rounds=50,
                                                   cv=self.cv, random_state=self.random_state), 
                      n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
        self.best_params = study.best_params

      elif self.model_name == 'RandomForestClassifier':
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(lambda trial: rf_objective(trial, self.model, X_train, y_train,
                                                  scoring=self.scoring,
                                                  cv=self.cv, random_state=self.random_state), 
                      n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
        self.best_params = study.best_params

      self.model = self.model.set_params(**self.best_params)
      self.logger.info('Train model using tuned parameters')
      self.model.fit(X_train, y_train)

    else:
      self.logger.info('Train model using default parameters')
      self.model.fit(X_train ,y_train)

    self.logger.info(f' Train model: {self.model_name} (Done) '.center(50, '='))
    self.logger.info('')
  
  def evaluate(self, X_test ,y_test):
    self.logger.info(f' Evaluate model: {self.model_name} '.center(50, '='))
    X_test = self.preprocessor.transform(X_test)
    y_pred = self.model.predict(X_test)
    y_prob = self.model.predict_proba(X_test)[:,1]
    
    # Evaluate model
    self.logger.info('evalute model')
    self.eval_result = evaluate_classification_model(self.model, self.out_feature, y_test, y_pred, y_prob)
    self.logger.info(f' Evaluate model: {self.model_name} (Done) '.center(50, '='))
    self.logger.info('')
    
    return self.eval_result

  def predict(self, input):
    input = input[self.in_feature]
    input = self.preprocessor.transform(input)
    return self.model.predict(input)
    
  def predict_proba(self, input):
    input = input[self.in_feature]
    input = self.preprocessor.transform(input)
    return self.model.predict_proba(input)

  def plot_confusion_matrix(self, title=f'Confusion Matrix'):
    fig = plt.figure(figsize=(10,6))
    cf_matrix = self.eval_result['cm']
    ax = plot_confusion_matrix(cf_matrix, title=title)
    return fig

  def plot_feature_importance(self, kind='barh'):
    fig, ax = plt.subplots()
    fi = self.eval_result['fi']
    if kind=='barh':
      fi.set_index('feature') \
        .sort_values(by='importance') \
        .plot(kind=kind, figsize=(10,6), ax=ax)
    elif kind=='bar':
      fi.set_index('feature') \
        .sort_values(by='importance') \
        .plot(kind=kind, figsize=(10,6), ax=ax)
    elif kind=='table':
      fi.style.background_gradient()
    return fig

  def plot_roc(self, title='Receiver operating characteristic (ROC)'):
    fig, ax = plt.subplots(figsize=(10, 6))
    fpr = self.eval_result['fpr']
    tpr = self.eval_result['tpr']
    auc = self.eval_result['auc']
    roc = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=auc
    )
    roc.plot(name=title, ax=ax)
    return fig

  def plot_pr(self, title='Precision Recall Curve (PR)'):
    fig, ax = plt.subplots(figsize=(10, 6))
    p = self.eval_result['precision']
    r = self.eval_result['recall']
    ap = self.eval_result['ap']
    pr = PrecisionRecallDisplay(
        recall=r,
        precision=p,
        average_precision=ap
    )
    pr.plot(name=title, ax=ax)
    return fig

  def summary_report(self):
    print("Classification Report")
    print(self.eval_result["classification_report"])
    print()
    print(f'Accuracy: {self.eval_result["accuracy_score"]}')
    print(f'Recall: {self.eval_result["recall_score"]}')
    print(f'Precision: {self.eval_result["precision_score"]}')
    print(f'F1 Score: {self.eval_result["f1_score"]}')
    print(f'AUC score: {self.eval_result["auc"]}')
    print(f'Average Precision score: {self.eval_result["ap"]}')

  def summary_result(self):
    self.summary_report()
    self.plot_feature_importance()
    self.plot_confusion_matrix()
    self.plot_roc()
    self.plot_pr()

  def save(self, save_path):
    self.logger.info(' Save result '.center(50, '='))
    self.logger.info(f'Save result at {save_path}')
    self.logger.info('Create directory')
    os.makedirs(save_path, exist_ok=True)

    # Save confusion matrix
    fig = self.plot_confusion_matrix()
    fig.savefig(os.path.join(save_path, 'confusion_matrix.png'), bbox_inches = 'tight')

    # Save feature importance
    fig = self.plot_feature_importance()
    fig.savefig(os.path.join(save_path, 'feature_importance.png'), bbox_inches = 'tight')

    # Save ROC
    fig = self.plot_roc()
    fig.savefig(os.path.join(save_path, 'roc_curve.png'), bbox_inches = 'tight')

    # Save PR
    fig = self.plot_pr()
    fig.savefig(os.path.join(save_path, 'pr_curve.png'), bbox_inches = 'tight')

    # Save result
    result = {'Accuracy': self.eval_result["accuracy_score"],
              'Recall': self.eval_result["recall_score"],
              'Precision': self.eval_result["precision_score"],
              'F1 Score': self.eval_result["f1_score"],
              'AUC score': self.eval_result["auc"],
              'Average Precision score': self.eval_result["ap"]}
    
    with open(os.path.join(save_path, 'result.json'), 'w') as f:
      json.dump(result, f)

    # Save application
    joblib.dump(self, os.path.join(save_path, 'pipeline.pkl'))
    self.logger.info(' Save result (Done) '.center(50, '='))
