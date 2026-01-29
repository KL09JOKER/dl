import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
 
data = load_iris() 
X = data.data 
y = data.target 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
 
model = Sequential() 
model.add(Dense(16, input_shape=(4,), activation='relu')) 
model.add(Dense(3, activation='softmax')) 
 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 
 
model.fit(X_train, y_train, epochs=1, batch_size=24) 
 
loss, accuracy = model.evaluate(X_test, y_test) 
print("Accuracy:", accuracy) 
