import numpy as np 
 
states = ['S0', 'S1'] 
actions = ['A0', 'A1'] 
 
rewards = { 
    ('S0','A0'): 5, 
    ('S0','A1'): 10, 
    ('S1','A0'): -1, 
    ('S1','A1'): 2 
} 
 
gamma = 0.9 
V = {'S0': 0, 'S1': 0} 
 
for _ in range(10): 
    V_new = V.copy() 
    for s in states: 
        values = [] 
        for a in actions: 
            values.append(rewards[(s,a)] + gamma * V[s]) 
        V_new[s] = max(values) 
    V = V_new 
 
print("State Values:") 
print(V) 
