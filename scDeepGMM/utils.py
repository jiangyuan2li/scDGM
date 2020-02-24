from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import torch.nn as nn

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.1)
        else:
            nn.init.constant_(param.data, 0)

def NMI(pred_label, true_label):
    return normalized_mutual_info_score(true_label, pred_label)

def ARI(pred_label, true_label):
	return adjusted_rand_score(true_label, pred_label)
