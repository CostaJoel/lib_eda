import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score,auc,\
    precision_score,\
    roc_auc_score,\
    roc_curve,\
    recall_score,\
    log_loss,\
    confusion_matrix,\
    classification_report,\
    accuracy_score


def curva_precision_recall_cv(clf, X, y, cv=3):
    probas = cross_val_predict(clf,
                               X,
                               y,
                               cv=cv,
                               method="predict_proba")

    precisions, recalls, thresholds = precision_recall_curve(y, probas[:, 1])

    f1s = []
    for i in thresholds:
        y_pred = (probas[:, 1] >= i)
        f1s.append(f1_score(y, y_pred))

    sns.set_style('whitegrid')
    plt.figure()
    ax = sns.lineplot(x=thresholds, y=precisions[:-1], color="b", label="precision")
    sns.lineplot(x=thresholds, y=recalls[:-1], color="g", label="recall", ax=ax)
    sns.lineplot(x=thresholds, y=f1s, color='red', label='f1', ax=ax)
    ax.figure.set_size_inches(8, 4)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_title('Curva Precision-Recall', fontsize=18)
    ax.set_xlabel('Limiares', fontsize=14)
    plt.show()

def curva_precision_recall(y, probas, formula_match = True):
    precisions, recalls, thresholds = precision_recall_curve(y, probas)
    pr_re_auc = auc(recalls, precisions)
    f1s = []
    conversoes = []

    for i in thresholds:
        y_pred = (probas >= i)
        f1s.append(f1_score(y, y_pred))
        conversoes.append(np.count_nonzero(y_pred == True) / y_pred.shape[0])
    if formula_match:
        thresholds_match = thresholds * 0.7 + 0.3
    else:
        thresholds_match = thresholds

    sns.set_style('whitegrid')
    plt.figure()
    ax = sns.lineplot(x=thresholds_match, y=precisions[:-1], color="b", label="precision")
    sns.lineplot(x=thresholds_match, y=recalls[:-1], color="g", label="recall", ax=ax)
    sns.lineplot(x=thresholds_match, y=conversoes, color='red', label='conversao', ax=ax)
    # sns.lineplot(x=thresholds_match, y=f1s* 0.7 + 3, color="red", label="f1", ax=ax)
    ax.figure.set_size_inches(8, 4)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    # ax.set_xticks(thresholds_match)
    ax.set_title('Curva Precision-Recall', fontsize=18)
    ax.set_xlabel('Limiares', fontsize=14)
    plt.show()

    plot_roc_curve(y, probas)

    plt.figure()
    plt.plot(recalls, precisions, label="AUC=" + str(round(pr_re_auc,2)))
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.axis([0, 1, 0, 1])
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.show()

def threshold_report_cv(clf, X, y, cv = 3):

    probas = cross_val_predict(clf,
                               X,
                               y,
                               cv=cv,
                               method="predict_proba")


    for i in np.arange(0, 1, 0.05):
        print("============= Thresh ", round(i, 2), "=================")
        y_pred = (probas[:, 1] >= i)

        print('Precision:', round(precision_score(y, y_pred) * 100, 2))
        print('Recall:', round(recall_score(y, y_pred) * 100, 2))
        print('F1:', round(f1_score(y, y_pred) * 100, 2))

        confusion = confusion_matrix(y, y_pred)
        print('confusion matrix\n', confusion)
        print(classification_report(y, y_pred))

        print('Log Loss:', round(log_loss(y, probas) * 100, 2))
        print('AUC:', round(roc_auc_score(y, probas[:, -1]) * 100, 2))


def threshold_report_table_cv(clf, X, y, cv = 3):
    probas = cross_val_predict(clf,
                               X,
                               y,
                               cv=cv,
                               method="predict_proba")
    resultado_total = []

    for treshold in np.arange(0, 1, 0.05):
        resultados = []
        y_pred = (probas[:,1] >= treshold)

        resultados.append(treshold)
        resultados.append((precision_score(y, y_pred) * 100))
        resultados.append((recall_score(y, y_pred) * 100))
        resultados.append((f1_score(y, y_pred) * 100))
        resultados.append(log_loss(y, probas[:,1]))
        resultados.append(roc_auc_score(y, probas[:, 1]))
        resultado_total.append(resultados)

    return pd.DataFrame(resultado_total, columns=['treshold', 'precision', 'recall', 'f1', 'log_loss', 'auc']).round(3)

def threshold_report(y, probas):
    for i in np.arange(0, 1, 0.05):
        print("============= Thresh ", round(i), "=================")
        y_pred = (probas >= i)

        print('Precision:', round(precision_score(y, y_pred) * 100, 2))
        print('Recall:', round(recall_score(y, y_pred) * 100, 2))
        print('F1:', round(f1_score(y, y_pred) * 100, 2))

        confusion = confusion_matrix(y, y_pred)
        print('confusion matrix\n', confusion)
        print(classification_report(y, y_pred))

    print('Log Loss:', round(log_loss(y, probas) * 100, 2))
    print('AUC:', round(roc_auc_score(y, probas) * 100, 2))


def threshold_report_escalada_match(y, probas):
    threshs_match = np.arange(0.3, 1, 0.025)
    probas_match = probas * 0.7 + 0.3

    for i in threshs_match:
        print("============= Thresh ", i, "=================")
        y_pred = (probas_match >= i)

        print('Precision:', round(precision_score(y, y_pred) * 100, 2))
        print('Recall:', round(recall_score(y, y_pred) * 100, 2))
        print('F1:', round(f1_score(y, y_pred) * 100, 2))

        confusion = confusion_matrix(y, y_pred)
        print('confusion matrix\n', confusion)
        print(classification_report(y, y_pred))

    print('Log Loss:', round(log_loss(y, probas) * 100, 2))
    print('AUC:', round(roc_auc_score(y, probas) * 100, 2))


def plot_roc_curve_cv(clf, X, y, cv=5):
    probas = cross_val_predict(clf,
                               X,
                               y,
                                cv=cv,
                               method="predict_proba")

    fpr, tpr, thresholds = roc_curve(y, probas[:,1])
    auc = roc_auc_score(y, probas[:,1])
    plt.plot(fpr, tpr, label="AUC=" + str(auc.round(2)))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.axis([0, 1, 0, 1])
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_curve(y, probas):
    fpr, tpr, thresholds = roc_curve(y, probas)
    auc = roc_auc_score(y, probas)
    plt.plot(fpr, tpr, label="AUC=" + str(auc.round(2)))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.axis([0, 1, 0, 1])
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_curve(y, probas):
    fpr, tpr, thresholds = roc_curve(y, probas)
    auc = roc_auc_score(y, probas)
    plt.plot(fpr, tpr, label="AUC=" + str(auc.round(2)))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.axis([0, 1, 0, 1])
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def threshold_report_table(y, probas, labels=('perf', 'nao_perf'), formula_match = True):
    if formula_match:
        threshs_match = np.arange(0.3, 1, 0.025)
        probas_match = probas * 0.7 + 0.3
    else:
        threshs_match = np.arange(0,1.05,0.05)
        probas_match = probas

    resultado_total = []
    precisions, recalls, thresholds = precision_recall_curve(y, probas)
    pr_re_auc = auc(recalls, precisions)
    for treshold in threshs_match:
        resultados = []
        y_pred = (probas_match >= treshold)
        resultados.append(treshold*100)
        resultados.append((precision_score(y, y_pred)))
        resultados.append((recall_score(y, y_pred)))
        # resultados.append((f1_score(y, y_pred) * 100))
        resultados.append(np.count_nonzero(y_pred == True))
        resultados.append(round((np.count_nonzero(y_pred == True) / y_pred.shape[0]), 2))
        resultados.append(np.count_nonzero(y[y_pred] == 1))
        resultados.append(np.count_nonzero(y[y_pred] == 0))
        resultados.append(round(np.count_nonzero(y[y_pred] == 1) / y.value_counts().loc[1], 2))
        resultados.append(round(np.count_nonzero(y[y_pred] == 0) / y.value_counts().loc[0], 2))
        resultados.append(pr_re_auc)
        resultados.append(roc_auc_score(y, probas))
        resultado_total.append(resultados)

    return pd.DataFrame(resultado_total, columns=['threshold', 'precision', 'recall',  'conversao', 'conversao(%)', labels[0], labels[1], labels[0] + "(%)", labels[1] + "(%)",'pr_re_auc', 'roc_auc']).round(3)




