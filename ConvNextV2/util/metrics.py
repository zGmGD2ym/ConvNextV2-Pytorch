import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix and return the matplotlib Figure."""
    figure = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_roc_curve_multiclass(labels_one_hot, preds_prob, class_names):
    """Plot multi-class ROC curve using one-vs-rest strategy."""
    num_classes = labels_one_hot.shape[1]
    plt.figure(figsize=(6, 6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_one_hot[:, i], preds_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")

    figure = plt.gcf()
    return figure


def plot_pr_curve_multiclass(labels_one_hot, preds_prob, class_names):
    """Plot multi-class Precision-Recall curve using one-vs-rest strategy."""
    num_classes = labels_one_hot.shape[1]
    plt.figure(figsize=(6, 6))

    for i in range(num_classes):
        precision_vals, recall_vals, _ = precision_recall_curve(labels_one_hot[:, i], preds_prob[:, i])
        pr_auc = auc(recall_vals, precision_vals)
        plt.plot(recall_vals, precision_vals, label=f'{class_names[i]} (AUC = {pr_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall')
    plt.legend(loc="upper right")

    figure = plt.gcf()
    return figure


def calculate_specificity(cm):
    """Calculate specificity for each class and return the mean."""
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificities.append(spec)
    return np.mean(specificities)
