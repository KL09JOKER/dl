states = ['S0', 'S1'] 
actions = ['A0', 'A1'] 
 
 
rewards = { 
    ('S0','A0'): 5, 
    ('S0','A1'): 10, 
    ('S1','A0'): -1, 
    ('S1','A1'): 2 
} 
 
transitions = { 
    ('S0','A0'): [('S0', 1.0)], 
    ('S0','A1'): [('S1', 1.0)], 
    ('S1','A0'): [('S0', 1.0)], 
    ('S1','A1'): [('S1', 1.0)] 
} 
 
gamma = 0.9   # Discount factor 
V = {'S0': 0, 'S1': 0}   
 
for _ in range(10): 
    V_new = V.copy() 
     
    for s in states: 
        action_values = [] 
         
        for a in actions: 
            total = 0 
            for next_state, prob in transitions[(s, a)]: 
                total += prob * (rewards[(s, a)] + gamma * V[next_state]) 
             
            action_values.append(total) 
         
        V_new[s] = max(action_values) 
     
    V = V_new 
 
print("Optimal State Values:") 
print(V) 
