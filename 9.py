import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from sklearn.preprocessing import MinMaxScaler 
 
data = np.array([[100],[102],[104],[106],[108],[110],[112],[114],[116],[118]]) 
 
scaler = MinMaxScaler() 
data = scaler.fit_transform(data) 
 
X = [] 
y = [] 
 
for i in range(len(data)-1): 
    X.append(data[i]) 
    y.append(data[i+1]) 
 
X = np.array(X).reshape(-1,1,1) 
y = np.array(y) 
 
model = Sequential() 
model.add(LSTM(20, input_shape=(1,1))) 
model.add(Dense(1)) 
 
model.compile(optimizer='adam', loss='mse') 
model.fit(X, y, epochs=200, verbose=0) 
 
pred = model.predict(X) 
pred = scaler.inverse_transform(pred) 
 
print("Predicted Stock Prices:") 
print(pred.flatten()) 
