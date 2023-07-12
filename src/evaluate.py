import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, \
                            ConfusionMatrixDisplay, confusion_matrix, \
                            RocCurveDisplay, roc_auc_score, roc_curve, \
                            PrecisionRecallDisplay, average_precision_score, precision_recall_curve\


def evaluate_classification_model(model, features, y_test, y_pred, y_prob):

  # Summary
  report = classification_report(y_test, y_pred)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  
  # ROC Curve and AUC
  fpr, tpr, _ = roc_curve(y_test, y_prob)
  auc = roc_auc_score(y_test, y_prob)

  # Precision-Recall Curve
  p, r, _ = precision_recall_curve(y_test, y_prob)
  ap = average_precision_score(y_test, y_prob)

  # Confusion matrix 
  cm = confusion_matrix(y_test, y_pred)

  # Feature importance
  fi = pd.DataFrame()
  fi['feature'] = features
  if hasattr(model, 'feature_importances_'):
    fi['importance'] = model.feature_importances_
  elif hasattr(model, 'coef_'):
    fi['importance'] = model.coef_.transpose()
  else:
    fi['importance'] = np.nan

  return {
      'model': model,
      'accuracy_score': accuracy,
      'precision_score': precision,
      'recall_score': recall,
      'f1_score': f1,
      'classification_report': report,
      'cm': cm,
      'auc': auc,
      'fpr': fpr,
      'tpr': tpr,
      'precision': p,
      'recall': r,
      'ap': ap,
      'fi': fi
  }
