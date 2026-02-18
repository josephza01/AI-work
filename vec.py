from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# โหลดข้อมูลชุด Digits
data = datasets.load_digits()

# แบ่งข้อมูลเป็น Training set และ Test set
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

# ปรับมาตรฐานข้อมูล (Feature Scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างและฝึกสอนโมเดล SVM
clfsvm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
clfsvm.fit(X_train, y_train)

# ทำนายผลและประเมินประสิทธิภาพ
anssvm = clfsvm.predict(X_test)
accsvm = accuracy_score(y_test, anssvm)

print('Accuracy = ', accsvm)
print(classification_report(y_test, anssvm))