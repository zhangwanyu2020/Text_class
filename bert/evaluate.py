import json
import numpy as np

def bool_to_value(pred_bool):
    predictions = []
    for i in range(len(pred_bool)):
        p_sigle = [0] * 95
        for j in range(len(pred_bool[i])):
            if pred_bool[i][j]:
                p_sigle[j] = 1
            else:
                p_sigle[j] = 0
        predictions.append(p_sigle)
    return predictions

def f1_np(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)
    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (possible_positives + 1e-8)
    macro_f1 = np.mean(2 * precision * recall / (precision + recall + 1e-8))

    """Micro_F1 metric.
    """
    precision = np.sum(true_positives) / np.sum(predicted_positives)
    recall = np.sum(true_positives) / np.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return micro_f1, macro_f1



def id_to_labeltext(id_list):
    label_list = []
    with open('label_id.json', encoding='utf-8') as f:
        line = f.readline()
        label_map = json.loads(line)
        reverse_label_map = {v:k for k,v in label_map.items()}
    for i in range(len(id_list)):
        label_sigle = ''
        for j in range(len(id_list[i])):
            if id_list[i][j]==1:
                l = reverse_label_map[j]
                label_sigle = label_sigle+' '+l
                label_sigle = label_sigle.strip()
        label_list.append(label_sigle)
    return label_list