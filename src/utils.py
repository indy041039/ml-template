import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.metrics import PrecisionRecallDisplay

def compare_feature_importance(list_result):
  df = pd.DataFrame()
  df['feature'] = list_result[0]['fi']['feature']
  for result in list_result:
    df[f"{result['model'].__class__.__name__}"] = result['fi']['importance']
  return df


def compare_roc(list_result):
  for result in list_result:
    model_name = result['model'].__class__.__name__
    plt.plot(result['fpr'], result['tpr'], label='%s ROC (area = %0.2f)' % (model_name, result['auc']))
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC)')
  plt.legend(loc="lower right")
  plt.show()  


def compare_pr(list_result):
  _, ax = plt.subplots(figsize=(10, 6))
  for result in list_result:
    model_name = result['model'].__class__.__name__
    display = PrecisionRecallDisplay(
      recall=result['recall'],
      precision=result['precision'],
      average_precision=result['ap'],
    )
    display.plot(ax=ax, name=model_name)
  plt.title('Precision Recall Curve')
  plt.legend(loc="upper right")
  plt.show()  


def plot_pr(model_name, r, p, ap):
  display = PrecisionRecallDisplay(
    recall=r,
    precision=p,
    average_precision=ap,
  )
  display.plot(name=f"Precision Recall Curve ({model_name})")

def compare_metric(list_result):
  result_dic = []
  for result in list_result:
    model_name = result['model'].__class__.__name__
    accuracy = result['accuracy_score']
    recall = result['recall_score']
    precision = result['precision_score']
    f1 = result['f1_score']
    auc = result['auc']
    result_dic.append({'model': model_name,
          'accuracy': accuracy,
          'recall': recall,
          'precision': precision,
          'f1': f1,
          'AUC': auc})
  return pd.DataFrame(result_dic)


def compare_model(list_result):
  compare_m = compare_metric(list_result)
  display(compare_m.style.background_gradient(axis=0))

  compare_fi = compare_feature_importance(list_result)
  display(compare_fi.style.background_gradient(axis=0))

  plt.figure(figsize=(10,6))
  compare_roc(list_result)

  plt.figure(figsize=(10,6))
  compare_pr(list_result)


def plot_confusion_matrix(cf_matrix, xyticks=[0, 1], title=''):
  group_counts = ['{0:0.0f}'.format(value) for value in
                  cf_matrix.flatten()]
  norm_cf_matrix = cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis]
  group_percentages = ['{0:.2%}'.format(value) for value in
                      norm_cf_matrix.flatten()]
  labels = [f'{v1}\n{v2}' for v1, v2 in
            zip(group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  cf_matrix = pd.DataFrame(cf_matrix, 
                           index=[f'Actual {tick}' for tick in xyticks],
                           columns=[f'Predicted {tick}' for tick in xyticks])
  fig = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
  fig.set_title(f'Confusion Matrix ({title})', fontsize = 13)
  return fig


def plot_roc(model_name, fpr, tpr, auc):
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (model_name, auc))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

  
def visualize_result(result):
  # Summary
  model_name = result["model"].__class__.__name__
  print(f'{model_name}')
  print()
  print(f'Accuracy Score: {result["accuracy_score"]}')
  print(f'Precision Score: {result["precision_score"]}')
  print(f'Recall Score: {result["recall_score"]}')
  print(f'F1-score Score: {result["f1_score"]}')
  print(f'AUC Score: {result["auc"]}')
  print()
  print(result["classification_report"])
  print()

  # Visualize Feature Importance
  display(result["fi"].sort_values('importance', ascending=False) \
                      .style.background_gradient(axis=0))
  
  # Visualize Confusion Matrix
  fig = plot_confusion_matrix(cf_matrix=result["cm"], title=model_name)
  plt.show()

  # Visualize ROC
  plot_roc(model_name, result["fpr"], result["tpr"], result["auc"])

  # Visualize PR
  plot_pr(model_name, result["recall"], result["precision"] ,result["ap"])

  plt.show()


def plot_class_probability(y_test, y_prob):
  plt.figure(figsize=(10,6))
  plt.hist(y_prob[y_test==0], bins=50, label='Negatives')
  plt.hist(y_prob[y_test==1], bins=50, label='Positives', alpha=0.7, color='r')
  plt.xlabel('Probability', fontsize=11)
  plt.ylabel('Count', fontsize=11)
  plt.legend(fontsize=8)
  plt.show()


def correlated_with_target(df, target, threshold=0.5):
  df = df.copy()
  # Pearson Correlation
  plt.figure(figsize=(12,10))
  cor = df.corr()

  #Correlation with output variable
  cor_target = abs(cor[target])

  #Selecting highly correlated features
  relevant_features = cor_target[cor_target > threshold]

  sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
  plt.show()

  return relevant_features


def correlated_between_features(X_train, threshold=0.8):
  df = X_train.copy()
  corr = df.corr()
  columns = np.full((corr.shape[0],), True, dtype=bool)
  for i in range(corr.shape[0]):
      for j in range(i+1, corr.shape[0]):
          if corr.iloc[i,j] >= threshold:
              if columns[j]:
                  columns[j] = False
  feature_correlated = df.columns[columns]
  return feature_correlated


def compute_weight(y):
  count_class = y.value_counts(normalize=True)
  count_negative = count_class[0]
  count_positive = count_class[1]
  weight = count_negative / count_positive
  return weight


def to_frame(input):
  input_df = pd.DataFrame([input])
  return input_df
