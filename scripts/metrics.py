import math
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def calculate_metric(pred_y, labels, pred_prob):
    """
    计算二分类模型的各种评估指标
    
    Args:
        pred_y: list, 预测的标签 [n_sample]
        labels: list, 真实标签 [n_sample] 
        pred_prob: list, 预测的概率 [n_sample]
    
    Returns:
        metric: dict, 包含各种指标的字典
        roc_data: list, [fpr, tpr, AUC]
        prc_data: list, [recall, precision, AP]
    """
    
    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # 计算混淆矩阵
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # 计算各种指标
    # ACC (Accuracy)
    ACC = float(tp + tn) / test_num

    # Precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # Recall/Sensitivity (SE)
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity (SP)
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC (Matthews Correlation Coefficient)
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # ROC和AUC
    fpr, tpr, thresholds = roc_curve(labels, pred_prob)
    AUC = auc(fpr, tpr)

    # PRC和AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob)
    PRC = average_precision_score(labels, pred_prob)

    # 将结果整理成字典，方便使用
    metric = {
        'ACC': ACC,
        'Precision': Precision, 
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'AUC': AUC,
        'PRC': PRC,  
        'MCC': MCC,
        'F1': F1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }
    
    return AUC, PRC, MCC, F1, ACC