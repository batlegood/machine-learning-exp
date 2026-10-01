import random
import numpy as np
from sklearn.linear_model import SGDRegressor


ROWS, COLS = 10, 10
START = (0, 0)
GOAL = (9, 9)


GRID = [
    ["A","o","o","o","#","o","o","o","o","o"],
    ["o","#","o","o","o","o","D","o","#","o"],
    ["o","o","o","#","o","o","o","o","o","o"],
    ["o","D","o","o","o","#","o","o","o","o"],
    ["o","o","#","o","o","o","o","D","o","o"],
    ["o","o","o","o","#","o","o","o","o","o"],
    ["o","o","o","o","o","o","#","o","o","o"],
    ["o","o","D","o","o","o","o","o","#","o"],
    ["o","o","o","o","o","D","o","o","o","o"],
    ["o","o","o","o","o","o","o","o","o","T"]
]

ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
ACTION_NAMES = ["Up","Down","Left","Right"]
NUM_ACTIONS = len(ACTIONS)
NUM_FEATURES = ROWS*COLS*NUM_ACTIONS


REWARD_MOVE = -1
REWARD_INVALID = -5
REWARD_WALL = -5
REWARD_DANGER = -10
REWARD_GOAL = 50

def cell_type(state):
    r,c = state
    return GRID[r][c]

def step(state, action):
    r = state[0] + ACTIONS[action][0]
    c = state[1] + ACTIONS[action][1]
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return state, REWARD_INVALID, False, "Invalid"
    cell = GRID[r][c]
    if cell == "#":
        return state, REWARD_WALL, False, "Wall"
    if cell == "D":
        return (r,c), REWARD_DANGER, False, "Danger"
    if cell == "T":
        return (r,c), REWARD_GOAL, True, "Goal"
    return (r,c), REWARD_MOVE, False, "Path"

def encode(state, action):
    features = np.zeros(NUM_FEATURES)
    idx = state[0]*COLS+state[1]
    features[idx*NUM_ACTIONS+action] = 1.0
    return features

def predict_q(model, state):
    features = np.array([encode(state,a) for a in range(NUM_ACTIONS)])
    return model.predict(features)

def train(episodes=1000, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
    rng = random.Random(42)
    model = SGDRegressor(loss="squared_error", penalty=None, fit_intercept=False,
                         learning_rate="constant", eta0=0.1, random_state=42)
    model.partial_fit(np.zeros((1,NUM_FEATURES)), np.array([0.0]))
    successes, rewards = 0, []
    for _ in range(episodes):
        state,total = START,0
        for _ in range(200):  # max steps
            if rng.random() < epsilon:
                action = rng.randrange(NUM_ACTIONS)
            else:
                q_values = predict_q(model,state)
                action = rng.choice(np.flatnonzero(q_values==q_values.max()).tolist())
            next_state,reward,terminated,ctype = step(state,action)
            target = reward if terminated else reward + gamma*predict_q(model,next_state).max()
            model.partial_fit(encode(state,action).reshape(1,-1), np.array([target]))
            state,total = next_state,total+reward
            if terminated:
                successes += 1
                break
        rewards.append(total)
        epsilon = max(epsilon_min, epsilon*epsilon_decay)

    
    state,path,steps = START,[START],[]
    total_eval_reward = 0
    for n in range(1,201):
        q_values = predict_q(model,state)
        action = int(np.argmax(q_values))
        next_state,reward,terminated,ctype = step(state,action)
        steps.append({"number":n,"state":state,"action":ACTION_NAMES[action],
                      "next_state":next_state,"cell":ctype,"reward":reward})
        path.append(next_state)
        total_eval_reward += reward
        state = next_state
        if terminated: break
    reached_goal = state==GOAL

    
    q_table=[]
    for r in range(ROWS):
        for c in range(COLS):
            pos=(r,c)
            if GRID[r][c]!="#" and GRID[r][c]!="T":
                q_table.append({"state":pos,"action_values":predict_q(model,pos).tolist()})

    return {
        "episodes":episodes,
        "successes":successes,
        "success_rate": round(successes/episodes*100,2),
        "final_average": round(sum(rewards[-100:])/len(rewards[-100:]),2),
        "epsilon_final": round(epsilon,2),
        "epsilon_initial": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "gamma": 0.95,
        "reached_goal":reached_goal,
        "path":path,
        "steps":steps,
        "q_table":q_table,
        "eval_reward": total_eval_reward,
        "eval_steps": len(steps)
    }
