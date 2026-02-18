import numpy as np 
import random 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam 
class GridWorld: 
    def __init__(self): 
        self.goal = [2, 2] 
    def reset(self): 
        self.pos = [0, 0] 
        return np.array(self.pos) / 2 
    def step(self, action): 
        if action == 0 and self.pos[0] > 0: 
            self.pos[0] -= 1 
        if action == 1 and self.pos[0] < 2: 
            self.pos[0] += 1 
        if action == 2 and self.pos[1] > 0: 
            self.pos[1] -= 1 
        if action == 3 and self.pos[1] < 2: 
            self.pos[1] += 1 
        reward = 10 if self.pos == self.goal else -1 
        done = self.pos == self.goal 
        return np.array(self.pos) / 2, reward, done 
model = Sequential([ 
    Dense(24, activation='relu', input_shape=(2,)), 
    Dense(24, activation='relu'), 
    Dense(4, activation='linear') 
]) 
model.compile(loss='mse', optimizer=Adam(0.001)) 
env = GridWorld() 
gamma = 0.9 
epsilon = 1.0 
episode_rewards = [] 
for episode in range(50):   
    state = env.reset().reshape(1, 2) 
    total_reward = 0 
    for step in range(20): 
        if random.random() < epsilon: 
            action = random.randint(0, 3) 
        else: 
            action = np.argmax(model.predict(state, verbose=0)[0]) 
        next_state, reward, done = env.step(action) 
        next_state = next_state.reshape(1, 2) 
        target = reward 
        if not done: 
            target += gamma * np.max(model.predict(next_state, verbose=0)[0]) 
        target_f = model.predict(state, verbose=0) 
        target_f[0][action] = target 
        model.fit(state, target_f, epochs=1, verbose=0) 
        state = next_state 
        total_reward += reward 
        if done: 
            break 
    epsilon *= 0.95 
    episode_rewards.append(total_reward) 
print("\nLast 3 Rewards:") 
print(episode_rewards[-3]) 
print(episode_rewards[-2]) 
print(episode_rewards[-1]) 




