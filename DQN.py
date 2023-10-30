#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque
import random


# In[2]:


import pandas as pd
import numpy as np
import random
import tensorflow as tf
from collections import deque

# Define hyperparameters
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001
TRAINING_EPISODES = 500

# Load data
data = pd.read_csv('grocery_data3.csv')

# Define state size and action size
state_size = len(data.columns) - 1
action_size = len(data['product_id'].unique())

# Define Q-Network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_dim=state_size, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))

# Define replay memory
memory = deque(maxlen=MEMORY_SIZE)

# Define function to select action
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    else:
        q_values = model.predict(state)
        return np.argmax(q_values[0])

# Define function to train Q-Network
def train_model():
    if len(memory) < BATCH_SIZE:
        return
    minibatch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + GAMMA * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

# Train Q-Network
epsilon = EPSILON
for episode in range(TRAINING_EPISODES):
    state = np.array(data.drop('product_id', axis=1).iloc[0])
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    for t in range(len(data)):
        action = select_action(state, epsilon)
        next_state = np.array(data.drop('product_id', axis=1).iloc[t])
        next_state = np.reshape(next_state, [1, state_size])
        reward = data['quantity'][t]
        done = False
        if t == len(data) - 1:
            done = True
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        train_model()
        if done:
            print('Episode {}/{}: Total Reward = {}, Epsilon = {:.2f}'.format(episode+1, TRAINING_EPISODES, total_reward, epsilon))
            break
    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY

# Evaluate Q-Network
def recommend_products(user_data, top_n=5):
    state = np.array(user_data)
    state = np.reshape(state, [1, state_size])
    q_values = model.predict(state)
    top_actions = np.argsort(q_values[0])[::-1][:top_n]
    recommended_products = [list(data['product_name'][data['product_id'] == a])[0] for a in top_actions]
    return recommended_products


# In[4]:


import pandas as pd
import numpy as np
import tensorflow as tf
# NCF
# Load data
data = pd.read_csv('grocery_data3.csv')

# Convert data to user-item matrix
user_item_matrix = pd.pivot_table(data, values='quantity', index='user_id', columns='product_id', fill_value=0)

# Define hyperparameters
NUM_USERS = len(user_item_matrix)
NUM_ITEMS = len(user_item_matrix.columns)
EMBEDDING_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# Split data into training and validation sets
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:]

# Define neural network model
input_user = tf.keras.layers.Input(shape=(1,))
embedding_user = tf.keras.layers.Embedding(input_dim=NUM_USERS, output_dim=EMBEDDING_SIZE)(input_user)
flatten_user = tf.keras.layers.Flatten()(embedding_user)

input_item = tf.keras.layers.Input(shape=(1,))
embedding_item = tf.keras.layers.Embedding(input_dim=NUM_ITEMS, output_dim=EMBEDDING_SIZE)(input_item)
flatten_item = tf.keras.layers.Flatten()(embedding_item)

dot_product = tf.keras.layers.Dot(axes=1)([flatten_user, flatten_item])
model = tf.keras.Model(inputs=[input_user, input_item], outputs=dot_product)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), loss='mean_squared_error')

# Define function to generate batches of training data
def generate_batches(data):
    num_batches = len(data) // BATCH_SIZE
    for i in range(num_batches):
        batch_data = data.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        user_ids = np.array(batch_data['user_id'])
        item_ids = np.array(batch_data['product_id'])
        ratings = np.array(batch_data['quantity'])
        yield [user_ids, item_ids], ratings

# Train model
for epoch in range(EPOCHS):
    for batch in generate_batches(train_data):
        inputs, targets = batch
        model.train_on_batch(inputs, targets)
    # Evaluate model on validation set
    val_loss = model.evaluate([val_data['user_id'], val_data['product_id']], val_data['quantity'], verbose=0)
    print(f"Epoch {epoch+1}/{EPOCHS}: Validation loss = {val_loss:.4f}")

# Recommend items to a user
def recommend_items(user_id, top_n=5):
    user_ratings = user_item_matrix.loc[user_id].values
    user_ratings = user_ratings.reshape(1, -1)
    item_indices = np.argsort(-user_ratings)[0][:top_n]
    item_ids = [user_item_matrix.columns[i] for i in item_indices]
    item_ratings = user_ratings[0, item_indices]
    return item_ids, item_ratings


# In[7]:


item_ids, item_ratings=recommend_items(1,5)
print(item_ids, item_ratings)


# In[12]:


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

def recommend_items(user_id, n=5):
    user_items = user_item_matrix.loc[user_id].values.reshape(1, -1)
    item_scores = model.predict(user_items)[0]
    item_indices = item_scores.argsort()[-n:][::-1]
    recommended_items = []
    for i in item_indices:
        if item_scores[i] > 0:
            recommended_items.append(items[i])
    return recommended_items


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
        

# In[13]:


model.save('recommendation_model.h5')
print("Model saved successfully!")


# $ pip install --no-binary=h5py h5py


def main(argv):
    print('DQN Args : ',argv)
    userid = int(argv[0])
    print('DQN Predicition : ',recommend_items(userid, 10, False))
    print('DQN Predicition Type: ',type(recommend_items(userid, 10, False)))
    return recommend_items(userid, 10, False)
    # args is a list of the command line args

if __name__ == "__main__":
   main(sys.argv[1:])
