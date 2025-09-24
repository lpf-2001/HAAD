from sklearn.metrics import confusion_matrix
import numpy as np

def get_metrics(y_test, predicted_labels):  # input shape is [Batch]  suchas y_test[0] = 10 means site 10
    cnf_matrix = confusion_matrix(y_test, predicted_labels)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/((TP+FN+1e-8))  # shape=[LabelNum]
    # Specificity or true negative rate
    TNR = TN/(TN+FP+1e-8)
    # Precision or positive predictive value
    PPV = TP/(TP+FP+1e-8)
    # print(TP)
    # print((TP+FP))
    # Negative predictive value
    NPV = TN/(TN+FN+1e-8)
    # Fall out or false positive rate
    FPR = FP/(FP+TN+1e-8)  # shape=[LabelNum]
    # False negative rate
    FNR = FN/(TP+FN+1e-8)
    # False discovery rate
    FDR = FP/(TP+FP+1e-8)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)  # shape=[LabelNum]
    F1 = 2*PPV*TPR/(PPV+TPR+1e-8)  # shape=[LabelNum]
    # entire acc
    overall_ACC = np.sum(TP) / np.sum(cnf_matrix)  # a float number, not equal to ACC.mean()

    return TPR,FPR,F1,ACC,overall_ACC