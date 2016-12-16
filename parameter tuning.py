import csv
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV

def ROCscore(p, testing_data):
	f = open("socres", 'w')
	for i in range(len(p)):
		f.write("%s %s %s\n"%(p[i][0], p[i][1], testing_data[i, 0]))
	f.close()
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
	return roc_auc_score(label, score)

def accuracy(p, results, testing_data):
	counter = 0
	error_0 = 0
	error_1 = 0
	valCtr = 0
	nonValCtr = 0
	
	for i in range(0, len(testing_data)):
		if testing_data[i, 0] == 0:
			nonValCtr += 1
		if testing_data[i, 0] == 1:
			valCtr += 1
		if testing_data[i, 0] == results[i]:
			counter += 1
		else:
			if testing_data[i, 0] == 0:
				error_0 += 1
			else:
				error_1 += 1
	return counter/len(testing_data)		
	#print("accuracy is ", counter/len(testing_data))
	#print("error on not val ", error_0/nonValCtr)
	#print("error on val ", error_1/valCtr)
	#target_names = ['class 0', 'class 1']
	#print(classification_report(testing_data[:, 0], results, target_names=target_names))

with open('selected_out.csv', 'rt') as csvfile:
	reader = list(csv.reader(csvfile))
	cols = len(reader[0])
	counetrs = [0] * cols
	headers = []
	for row_number in range(0, len(reader)):
		row = reader[row_number]
		if row_number == 0:
			headers = row
		for i in range(1, len(row)):
			if row[i] == '':
				reader[row_number][i] = np.nan
				counetrs[i] += 1
for i in range(0, cols):
	if counetrs[i] > 30000:
		print(headers[i])
features = np.array(reader)
features = features[1:, :]
np.random.shuffle(features)
imp = preprocessing.Imputer(missing_values = 'NaN', strategy = 'median', verbose = 0)
imp.fit(features)
features = imp.transform(features)

#five fold validation
data = []
length = int(len(features)/5)
for i in range(0, 5):
	data.append(features[i * length : (i + 1) * length, :])
	
#decision tree parameter tuning

print("decision tree")
for i in range(40, 161, 40):
	print("max depth is", i)
	accSum = 0
	precisionSum = 0
	recallSum = 0
	scoreSum = 0
	roc = 0
	for j in range(0, 5):
		training_data = []
		testing_data = data[j]
		for k in range(0, 5):
			if k != j:
				training_data += data[k].tolist()
		training_data = np.array(training_data)
		clf = tree.DecisionTreeClassifier(max_depth = i)
		X = training_data[:, 1:]
		Y = training_data[:, 0]
		clf.fit(X, Y)
		results = clf.predict(testing_data[:, 1:])
		p = clf.predict_proba(testing_data[:, 1:])
		accSum += accuracy(p, results, testing_data)
		precision, recall, score, support = precision_recall_fscore_support(testing_data[:, 0], results, average = 'binary')
		roc += ROCscore(p, testing_data)
		precisionSum += precision
		recallSum += recall
		scoreSum += score
	accSum = accSum/5
	precisionSum = precisionSum/5
	recallSum = recallSum/5
	scoreSum = scoreSum/5
	roc = roc/5
	print("accuracy is %.3f"%accSum)
	print("precision %.3f"%precisionSum, "recall %.3f"% recallSum, "score %.3f"% scoreSum, "ROC_AUC %.3f"%roc)


#random forest parameter tuning
print("random forest")
for i in range(16, 65, 8):
	for j in range(16, 129, 16):
		print("number of estimators is", i, " max_depth is ", j)
		accSum = 0
		precisionSum = 0
		recallSum = 0
		scoreSum = 0
		roc = 0
		for k in range(0, 5):
			training_data = []
			testing_data = data[k]
			for l in range(0, 5):
				if l != k:
					training_data += data[l].tolist()
			training_data = np.array(training_data)
			clf = RandomForestClassifier(n_estimators=i, max_depth=j, min_samples_split=2\
										,criterion = "entropy")
			X = training_data[:, 1:]
			Y = training_data[:, 0]
			clf.fit(X, Y)
			results = clf.predict(testing_data[:, 1:])
			p = clf.predict_proba(testing_data[:, 1:])
			accSum += accuracy(p, results, testing_data)
			precision, recall, score, support = precision_recall_fscore_support(testing_data[:, 0], results, average = 'binary')
			roc += ROCscore(p, testing_data)
			precisionSum += precision
			recallSum += recall
			scoreSum += score
		accSum = accSum/5
		precisionSum = precisionSum/5
		recallSum = recallSum/5
		scoreSum = scoreSum/5
		roc = roc/5
		print("accuracy is %.3f"%accSum)
		print("precision %.3f"%precisionSum, "recall %.3f"% recallSum, "score %.3f"% scoreSum, "ROC_AUC %.3f"%roc)
		

#gradient boosting parameter tuning
print("GradientBoostingClassifier")
for i in range(100, 1001, 200):
	print("number of iterations is: ", i)
	accSum = 0
	precisionSum = 0
	recallSum = 0
	scoreSum = 0
	roc = 0
	for k in range(0, 5):
		training_data = []
		testing_data = data[k]
		for l in range(0, 5):
			if l != k:
				training_data += data[l].tolist()
		training_data = np.array(training_data)
		clf = GradientBoostingClassifier(n_estimators=i, learning_rate=1.0, max_depth=1, random_state=0)
		X = training_data[:, 1:]
		Y = training_data[:, 0]
		clf.fit(X, Y)
		results = clf.predict(testing_data[:, 1:])
		p = clf.predict_proba(testing_data[:, 1:])
		accSum += accuracy(p, results, testing_data)
		precision, recall, score, support = precision_recall_fscore_support(testing_data[:, 0], results, average = 'binary')
		roc += ROCscore(p, testing_data)
		precisionSum += precision
		recallSum += recall
		scoreSum += score
	accSum = accSum/5
	precisionSum = precisionSum/5
	recallSum = recallSum/5
	scoreSum = scoreSum/5
	roc = roc/5
	print("accuracy is %.3f"%accSum)
	print("precision %.3f"%precisionSum, "recall %.3f"% recallSum, "score %.3f"% scoreSum, "ROC_AUC %.3f"%roc)



#logistic regression parameter tuning
print("LogisticRegressionCV")
i = 1
while i <= 1000:
	print("C is: ", i)
	i = i * 10
	accSum = 0
	precisionSum = 0
	recallSum = 0
	scoreSum = 0
	roc = 0
	for k in range(0, 5):
		training_data = []
		testing_data = data[k]
		for l in range(0, 5):
			if l != k:
				training_data += data[l].tolist()
		training_data = np.array(training_data)
		clf = LogisticRegressionCV(Cs = i)
		X = training_data[:, 1:]
		Y = training_data[:, 0]
		clf.fit(X, Y)
		results = clf.predict(testing_data[:, 1:])
		p = clf.predict_proba(testing_data[:, 1:])
		accSum += accuracy(p, results, testing_data)
		precision, recall, score, support = precision_recall_fscore_support(testing_data[:, 0], results, average = 'binary')
		roc += ROCscore(p, testing_data)
		precisionSum += precision
		recallSum += recall
		scoreSum += score
	precisionSum = precisionSum/5
	recallSum = recallSum/5
	scoreSum = scoreSum/5
	roc = roc/5
	print("accuracy is %.3f"%accSum)
	print("precision %.3f"%precisionSum, "recall %.3f"% recallSum, "score %.3f"% scoreSum, "ROC_AUC %.3f"%roc)

					



