import numpy as np 
 
X = np.array([[2,1], [1,-1], [3,2], [0,0]]) 
y = np.array([1, 0, 1, 0]) 
 
w = np.zeros(X.shape[1]) 
b = 0 
lr = 0.1 
 
def step(x): 
    return 1 if x >= 0 else 0 
 
for epoch in range(10): 
    for i in range(len(X)): 
        y_pred = step(np.dot(X[i], w) + b) 
        error = y[i] - y_pred 
        w += lr * error * X[i] 
        b += lr * error 
 
print("Weights:", w) 
print("Bias:", b) 
print("Predictions:") 
for i in range(len(X)): 
    print(X[i], "->", step(np.dot(X[i], w) + b)) 
