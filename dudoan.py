import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Lấy dữ liệu thực tế
symbol = 'AAPL'
data = yf.download(symbol, start='2020-01-01', end='2025-04-01')

# Kiểm tra nếu dữ liệu rỗng
if data.empty:
    raise ValueError("Không có dữ liệu, hãy kiểm tra lại mã cổ phiếu hoặc ngày.")

# Chỉ lấy cột Close Price
data = data[['Close']].copy()
data['Date'] = data.index

# Chuyển đổi ngày sang số nguyên (để dùng trong mô hình)
data['DateOrdinal'] = data['Date'].map(lambda x: x.toordinal())

# Chia tập dữ liệu
X = data[['DateOrdinal']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Vẽ biểu đồ
plt.figure(figsize=(10,6))
plt.plot(data['Date'][-len(y_test):], y_test, label='Giá thực tế', color='blue')
plt.plot(data['Date'][-len(y_test):], y_pred, label='Dự đoán', color='red', linestyle='--')
plt.xlabel('Ngày')
plt.ylabel('Giá cổ phiếu')
plt.title(f'Dự đoán xu hướng giá cổ phiếu của {symbol}')
plt.legend()
plt.xticks(rotation=45)
plt.show()
