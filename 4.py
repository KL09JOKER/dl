import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
 
data = np.array([10,12,14,16,18,20,22,24,26,28]) 
X = [] 
y = [] 
 
for i in range(len(data)-1): 
    X.append(data[i]) 
    y.append(data[i+1]) 
 
X = np.array(X).reshape(-1,1,1) 
y = np.array(y) 
 
model = Sequential() 
model.add(LSTM(10, input_shape=(1,1))) 
model.add(Dense(1)) 
 
model.compile(optimizer='adam', loss='mse') 
model.fit(X, y, epochs=200, verbose=0) 
 
pred = model.predict(X) 
print("Predicted Values:") 
print(pred.flatten()) 
