from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

def NMI(pred_label, true_label):
	return normalized_mutual_info_score(true_label, pred_label)

def ARI(pred_label, true_label):
	return adjusted_rand_score(true_label, pred_label)
