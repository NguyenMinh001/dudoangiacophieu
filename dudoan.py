import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


symbol = input("Nhap ma co phieu (vi du: AAPL): ").upper()
start_date = input("Nhap ngay bat dau (yyyy-mm-dd): ")
end_date = input("Nhap ngay ket thuc (yyyy-mm-dd): ")

data = yf.download(symbol, start=start_date, end=end_date)


if data.empty:
    raise ValueError("Không có dữ liệu, hãy kiểm tra lại mã cổ phiếu hoặc ngày.")


data = data[['Close']].copy()
data['Date'] = data.index


data['DateOrdinal'] = data['Date'].map(lambda x: x.toordinal())


X = data[['DateOrdinal']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')


plt.figure(figsize=(10,6))
plt.plot(data['Date'][-len(y_test):], y_test, label='Giá thực tế', color='blue')
plt.plot(data['Date'][-len(y_test):], y_pred, label='Dự đoán', color='red', linestyle='--')
plt.xlabel('Ngày')
plt.ylabel('Giá cổ phiếu')
plt.title(f'Dự đoán xu hướng giá cổ phiếu của {symbol}')
plt.legend()
plt.xticks(rotation=45)
plt.show()
