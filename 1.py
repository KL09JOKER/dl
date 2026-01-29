def step(x): 
    return 1 if x >= 0 else 0 
 
def perceptron(x1, x2, w1, w2, b): 
    total = x1*w1 + x2*w2 + b 
    return step(total) 
 
w1, w2, b = 1, 1, -1.5 
print("AND Gate Results:") 
for x1 in [0, 1]: 
    for x2 in [0, 1]: 
        print(f"{x1} AND {x2} = {perceptron(x1, x2, w1, w2, b)}") 
 
w1, w2, b = 1, 1, -0.5 
print("\nOR Gate Results:") 
for x1 in [0, 1]: 
    for x2 in [0, 1]: 
        print(f"{x1} OR {x2} = {perceptron(x1, x2, w1, w2, b)}")  
 
import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
 
X = np.array([[0,0],[0,1],[1,0],[1,1]]) 
y = np.array([0,1,1,0])   
 
model = Sequential() 
model.add(Dense(4, input_dim=2, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))            
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
 
model.fit(X, y, epochs=500, verbose=0) 
 
predictions = model.predict(X) 
print("\nPattern Classification (XOR) Predictions:") 
for i, pred in enumerate(predictions): 
    print(f"Input: {X[i]} -> Predicted Output: {round(pred[0])}")
