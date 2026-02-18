from sklearn.linear_model import LinearRegression
import numpy as np

# 1. เตรียมข้อมูลสอน (Training Data)
# X คือ โจทย์ (เลข 1, 2, 3, 4, 5)
X = np.array([[1], [2], [3], [4], [5]]) 
# y คือ เฉลย (แม่ 2: 2, 4, 6, 8, 10)
y = np.array([2, 4, 6, 8, 10])

# 2. สร้างโมเดลสมองกล
model = LinearRegression()

# 3. สอนโมเดล (Training)
# บอกมันว่า "ถ้าเห็น X แบบนี้ คำตอบคือ y นะ"
model.fit(X, y)

# 4. ทดลองทาย (Prediction)
# ลองถามมันว่า "ถ้าเลข 10 ล่ะ คำตอบควรเป็นเท่าไหร่?"
prediction = model.predict([[10]])

print(f"ผลลัพธ์ที่ AI ทาย: {prediction[0]}") 
# ผลลัพธ์ควรจะออกมาใกล้เคียง 20.0 มากๆ