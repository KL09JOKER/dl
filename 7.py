import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
 
states = np.array([[0,0],[0,1],[1,0],[1,1]]) 
actions = np.array([[1,0],[0,1],[0,1],[1,0]]) 
 
model = Sequential() 
model.add(Dense(16, input_dim=2, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(2, activation='softmax')) 
 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 
 
model.fit(states, actions, epochs=200, verbose=0) 
 
pred = model.predict(states) 
print("Agent Actions:") 
print(pred)
