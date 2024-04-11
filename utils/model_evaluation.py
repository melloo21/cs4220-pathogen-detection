import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import chain
from typing import Union, Any
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def _read_model(
    model_name:str,
    filepath:str
):
    model_name = model_name if ".joblib" in model_name else f"{model_name}.joblib"
    file = f"{filepath}/{model_name}"
    model = joblib.load(file)   

    return model

def _draw_confusion(    
    data_set:tuple,
    model_name:str,
    filepath:str,
    data_type:str
):
    x_val , y_true = data_set
    model = _read_model(
        model_name=model_name,
        filepath=filepath
    )
    
    y_pred = model.predict(x_val)
    # Returns precision/recall/f1 score

    # 'micro':
    # Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro':
    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted':
    # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    macro_score = precision_recall_fscore_support(y_true, y_pred, average='macro')[0:3]
    micro_score = precision_recall_fscore_support(y_true, y_pred, average='micro')[0:3]

    print(" Macro Precision : %5.2f, Recall : %5.2f, F1 : %5.2f" %  macro_score)
    print(" Micro Precision : %5.2f, Recall : %5.2f, F1 : %5.2f" %  micro_score)

    cm = confusion_matrix(y_true, y_pred)

    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(f'Confusion Matrix for {data_type}'); 
    
    plt.show()

def performance_evaluate(
    train_dataset:tuple,
    valid_dataset:tuple,
    model_name:str,
    filepath:str
    ):
    """
        Summary 
        Args:
        data_set:tuple -- (x_values, y_values) in numpy array format
        model_name:str  -- model file name to extract      
    """

    # Init
    _draw_confusion(
        data_set=train_dataset,
        model_name=model_name,
        filepath=filepath,
        data_type="Train data"       
    )
    _draw_confusion(
        data_set=valid_dataset,
        model_name=model_name,
        filepath=filepath,
        data_type="Valid data"       
    )

    return

def plot_auc_roc(
    valid_dataset:tuple,
    model_name:str,
    filepath:str    
):
    x_valid , y_valid = valid_dataset
    model = _read_model(
        model_name=model_name,
        filepath=filepath
    )
    model_pred_prob= model.predict_proba(x_valid)
    preds = model_pred_prob[:,1]
    fpr, tpr, threshold = roc_curve(y_valid, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def jaccard_index_per_patient(patient_id:str, preds):
    # Generic filename
    df_true = pd.read_csv('test_data/patient{}_labels.csv'.format(patient_id))
    tp, fp, tp_fn = 0, 0, df_true['labels'].shape[0]
    print('my predition(s) for patient {}:'.format(patient_id))
    print(preds)
    print('true pathogen')
    print(df_true['labels'].values)
    # if don't predict any pathogen, it means there is only decoy in the test dataset (your prediction)
    if len(preds) == 0:
        preds = ['decoy']
    for item in np.unique(preds):
        if item in df_true['labels'].values:
            tp += 1
        else:
            fp += 1
    #you have to predict all labels correctly, but you are penalized for any false positive
    return tp / (tp_fn + fp) , preds, df_true['labels'].values

def get_all_jaccard_index( model:Any, label_encoder:Any ,num_patients:int=10, threshold:float=0.95):

    all_jaccard_index = []
    all_pred = []
    all_true = []
    for patient_id in range(num_patients):
        print('predicting for patient {}'.format(patient_id))

        with open('test_data/patient{}_6mers.npy'.format(patient_id), 'rb') as read_file:
            df_test = np.load(read_file)

        # regr.predict relies on argmax, thus predict to every single read and you will end up with many false positives
        y_pred = model.predict(df_test)

        # we can use regr.predict_proba to find a good threshold and predict only for case where the model is confident.
        # here I apply 0.95 as the cutoff for my predictions, let's see how well my model will behave...
        y_predprob = model.predict_proba(df_test)

        # we get only predictions larger than the threshold and if there is more than one, we take the argmax again
        final_predictions = label_encoder.inverse_transform(
                                np.unique([np.argmax(item) for item in y_predprob if len(np.where(item >= threshold)[0]) >= 1]
                            ))

        # my pathogens dectected, decoy will be ignored
        final_predictions = [item for item in final_predictions if item !='decoy']

        ji, pred_pathogen, true_pathogen = jaccard_index_per_patient(patient_id, final_predictions)
        print('Jaccard index: {}'.format(ji))
        all_jaccard_index.append(ji)    
        all_pred.append(pred_pathogen)
        all_true.append(true_pathogen)

    return all_jaccard_index, flatten(all_pred), flatten(all_true)

def flatten(original_list:list):
    return list(chain.from_iterable(original_list))

def get_all_jaccard_index_with_transformation( model:Any, label_encoder:Any , x_transformer:Any, num_patients:int=10, threshold:float=0.95):

    all_jaccard_index = []
    all_pred = []
    all_true = []
    for patient_id in range(num_patients):
        print('predicting for patient {}'.format(patient_id))

        with open('test_data/patient{}_6mers.npy'.format(patient_id), 'rb') as read_file:
            df_test = np.load(read_file)

        # regr.predict relies on argmax, thus predict to every single read and you will end up with many false positives
        transformed_data =x_transformer.fit_transform(df_test)
        y_pred = model.predict(transformed_data)

        # we can use regr.predict_proba to find a good threshold and predict only for case where the model is confident.
        # here I apply 0.95 as the cutoff for my predictions, let's see how well my model will behave...
        y_predprob = model.predict_proba(transformed_data)

        # we get only predictions larger than the threshold and if there is more than one, we take the argmax again
        final_predictions = label_encoder.inverse_transform(
                                np.unique([np.argmax(item) for item in y_predprob if len(np.where(item >= threshold)[0]) >= 1]
                            ))

        # my pathogens dectected, decoy will be ignored
        final_predictions = [item for item in final_predictions if item !='decoy']

        ji, pred_pathogen, true_pathogen = jaccard_index_per_patient(patient_id, final_predictions)
        print('Jaccard index: {}'.format(ji))
        all_jaccard_index.append(ji)    
        all_pred.append(pred_pathogen)
        all_true.append(true_pathogen)

    return all_jaccard_index, flatten(all_pred), flatten(all_true)