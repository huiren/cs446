import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
file = open("socres", "r")
data = file.readlines()
file.close()

label = []
score = []

for line in data:
    line = line.strip()
    line = line.split(" ")
    score.append(float(line[1]))
    label.append(int(float(line[2])))
label = np.array(label)
score = np.array(score)

fpr, tpr, thresholds = roc_curve(label, score, pos_label=1)
print(roc_auc_score(label, score))