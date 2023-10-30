import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque
import random

# Load data
data = pd.read_csv('grocery_data3.csv')

# Convert data to user-item matrix
user_item_matrix = pd.pivot_table(data, values='quantity', index='user_id', columns='product_id', fill_value=0)

# Define hyperparameters
NUM_USERS = len(user_item_matrix)
NUM_ITEMS = len(user_item_matrix.columns)
EMBEDDING_SIZE = 32
BATCH_SIZE = 64
EPISODES = 100
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000

# Split data into training and validation sets
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:]

# Define neural network model
input_state = tf.keras.layers.Input(shape=(NUM_ITEMS,))
dense1 = tf.keras.layers.Dense(128, activation='relu')(input_state)
dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
output = tf.keras.layers.Dense(NUM_ITEMS, activation='linear')(dense2)
model = tf.keras.Model(inputs=input_state, outputs=output)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), loss='mean_squared_error')

# Define replay buffer
memory = deque(maxlen=MEMORY_SIZE)

# Define function to select an action
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return random.randrange(NUM_ITEMS)
    else:
        return np.argmax(model.predict(state))

# Define function to train the model
def train_model():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states = np.zeros((BATCH_SIZE, NUM_ITEMS))
    next_states = np.zeros((BATCH_SIZE, NUM_ITEMS))
    actions, rewards, done = [], [], []
    for i in range(BATCH_SIZE):
        states[i] = batch[i][0]
        actions.append(batch[i][1])
        rewards.append(batch[i][2])
        next_states[i] = batch[i][3]
        done.append(batch[i][4])
    targets = model.predict(states)
    next_q_values = model.predict(next_states)
    for i in range(BATCH_SIZE):
        if done[i]:
            targets[i][actions[i]] = rewards[i]
        else:
            targets[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])
    model.fit(states, targets, epochs=1, verbose=0)

# Train model
for episode in range(EPISODES):
    state = user_item_matrix.loc[random.choice(user_item_matrix.index)].values.reshape(1, -1)
    total_reward = 0
    epsilon = max(EPSILON_MIN, EPSILON_DECAY * EPSILON)
    for t in range(NUM_ITEMS):
        action = select_action(state, epsilon)
        reward = user_item_matrix.iloc[:, action].sum()
        next_state = np.copy(state)
        next_state[:, action] = reward
        done = t == NUM_ITEMS - 1
        memory.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state
        train_model()
        if done:
            break
    print(f"Episode {episode+1}/{EPISODES}: Total reward =  {total_reward}")
    if (episode+1) % 10 == 0:
        print("Recommended items for user 1:", recommend_items(1))

def recommend_items(user_id, n=5):
    user_items = user_item_matrix.loc[user_id].values.reshape(1, -1)
    item_scores = model.predict(user_items)[0]
    item_indices = item_scores.argsort()[-n:][::-1]
    recommended_items = []
    for i in item_indices:
        if item_scores[i] > 0:
            recommended_items.append(items[i])
    return recommended_items

# print(recommend_items(1,10))


def main(argv):
    print('Args : ',argv)
    userid = int(argv[0])
    print('DQN Predicition : ',recommend_items(userid, 10, False))
    print('DQN Predicition Type: ',type(recommend_items(userid, 10, False)))
    return recommend_items(userid, 10, False)
    # args is a list of the command line args

if __name__ == "__main__":
   main(sys.argv[1:])