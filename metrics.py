


def all_metrics(y_true, y_pred, heatmap = True):
	import os
	try:
		os.system('pip install scikit-learn seaborn')
    except Exception as e:
    	print(e)
    	
    from sklearn.metrics import f1_score, precision_score, recall_score,roc_auc_score, confusion_matrix
    import seaborn as sns
    
    print('f1 score: ',f1_score(y_true, y_pred))
    print( 'precision score: ', precision_score(y_true, y_pred))
    print('recall score: ',recall_score(y_true,y_pred))
    auc = roc_auc_score(y_true, y_pred)
    print('AUC: %.3f' % auc)
    if heatmap:
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True,cmap='Blues', fmt='g')
    else:
        print(confusion_matrix(y_true, y_pred))


# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
    -----
    y_true = true labels in the form of a 1D array
    y_pred = predicted labels in the form of a 1D array

    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
    return model_results
main():



if __name__=="__main__":
	
	main()
