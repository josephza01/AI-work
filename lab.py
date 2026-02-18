from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np


data = datasets.load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
knn_prob = knn.predict_proba(X_test)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=None)
dt.fit(X_train, y_train)
dt_prob = dt.predict_proba(X_test)

answer = knn.predict(X_test)
answer_dt = dt.predict(X_test)

accuracy_score_knn = accuracy_score(y_test, answer)
accuracy_score_dt = accuracy_score(y_test, answer_dt)
print("KNN Classifier Accuracy:", accuracy_score_knn)
print("Decision Tree Classifier Accuracy:", accuracy_score_dt)

per_recall_knn = classification_report(y_test, answer)
per_recall_dt = classification_report(y_test, answer_dt)
print("KNN Classifier Classification Report:\n", per_recall_knn)
print("Decision Tree Classifier Classification Report:\n", per_recall_dt)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

fpr_knn = dict()
tpr_knn = dict()
roc_auc_knn = dict()
fpr_dt = dict()
tpr_dt = dict()
roc_auc_dt = dict()

for i in range(n_classes):
    fpr_knn[i], tpr_knn[i], _ = roc_curve(y_test_bin[:, i], knn_prob[:, i])
    roc_auc_knn[i] = auc(fpr_knn[i], tpr_knn[i])
    
    fpr_dt[i], tpr_dt[i], _ = roc_curve(y_test_bin[:, i], dt_prob[:, i])
    roc_auc_dt[i] = auc(fpr_dt[i], tpr_dt[i])

fpr_knn["micro"], tpr_knn["micro"], _ = roc_curve(y_test_bin.ravel(), knn_prob.ravel())
roc_auc_knn["micro"] = auc(fpr_knn["micro"], tpr_knn["micro"])

fpr_dt["micro"], tpr_dt["micro"], _ = roc_curve(y_test_bin.ravel(), dt_prob.ravel())
roc_auc_dt["micro"] = auc(fpr_dt["micro"], tpr_dt["micro"])

print("KNN Classifier AUC (micro-average):", roc_auc_knn["micro"])
print("Decision Tree Classifier AUC (micro-average):", roc_auc_dt["micro"])

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn["micro"], tpr_knn["micro"], color='red', label='KNN Classifier (AUC = {:.2f})'.format(roc_auc_knn["micro"]))
plt.plot(fpr_dt["micro"], tpr_dt["micro"], color='blue', label='Decision Tree Classifier (AUC = {:.2f})'.format(roc_auc_dt["micro"]))
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison (Micro-Average)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.show()