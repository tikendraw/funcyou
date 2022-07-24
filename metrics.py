


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


main():



if __name__=="__main__":
	
	main()
