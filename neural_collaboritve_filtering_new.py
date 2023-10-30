#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data = pd.read_csv("grocery_data2.csv")


# In[ ]:


data.head(5)


# In[ ]:


data['purchased'] = [1] * len(data)


# In[ ]:


data.head(5)


# In[ ]:


# categorising the features into sparse/dense feature set 

# dense features are quantitative in nature
dense_features = ['quantity', 'age', 'price',]

#sparse features are categorical in nature
sparse_features = ['user_id', 'product_id', 'gender', 'location', 'product_name', 'category',  'brand', 'day_of_week', 'time_of_day']

# data imputation for missing values
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)


# In[ ]:


# encoding function
def encoding(data,feat,encoder):
    data[feat] = encoder.fit_transform(data[feat])
# encoding for categorical features
[encoding(data,feat,LabelEncoder()) for feat in sparse_features]
# Using normalization for dense feature
# mms = MinMaxScaler(feature_range=(0,1))
# data[dense_features] = mms.fit_transform(data[dense_features])


# In[ ]:


data['price'] = (data['price'] * 1000 ).astype(int)


# In[ ]:


data


# In[ ]:


items_lookup = {}
users_lookup = {}

for i in range(len(data)):
    items_lookup[data['product_id'].iloc[i]] = data.iloc[i][['category','price','brand']]
    users_lookup[data['user_id'].iloc[i]] = data.iloc[i][['age','location','gender']]


# In[ ]:





# In[ ]:


data['rank_latest'] = data.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)

train_df = data[data['rank_latest'] != 1]
test_df = data[data['rank_latest'] == 1]


# drop columns that we no longer need
train_df = train_df.drop(['timestamp','weather'], axis = 1)
test_df = test_df.drop(['timestamp','weather'], axis = 1)


# In[ ]:


# train_df['user_id'].unique()


# In[ ]:


# Get a list of all movie IDs
all_productIds = data['product_id'].unique()

# Placeholders that will hold the training data
_userid, _itemid, _age, _gender, _location, _category, _price, _brand, _purchased = [], [], [], [], [], [], [], [], []

# This is the set of items that each user has interaction with
user_items_dataset = set(zip(train_df['user_id'], train_df['product_id'], \
                         train_df['age'], train_df['gender'], \
                        train_df['location'], train_df['category'], \
                         train_df['price'], train_df['brand'], train_df['purchased']))

user_item_set = set(zip(train_df['user_id'], train_df['product_id']))

# 4:1 ratio of negative to positive samples
num_negatives = 4

for (user, item, age, gender, location, category, price, brand, purchased) in user_items_dataset:
    
    _userid.append(user)
    _itemid.append(item)
    _age.append(age)
    _gender.append(gender)
    _location.append(location)
    _category.append(category)
    _price.append(price)
    _brand.append(brand)
    _purchased.append(1)
    
    for _ in range(num_negatives):
        # randomly select an item
        negative_item = np.random.choice(all_productIds)
        # check that the user has not interacted with this item
        while (user, negative_item) in user_item_set:
            negative_item = np.random.choice(all_productIds)
            
        _userid.append(user)
        _itemid.append(negative_item)
        _age.append(age)
        _gender.append(gender)
        _location.append(location)
        _category.append(items_lookup[negative_item]['category'])
        _price.append(items_lookup[negative_item]['price'])
        _brand.append(items_lookup[negative_item]['brand'])
        _purchased.append(0)


# In[4]:


import torch
from torch.utils.data import Dataset

class ItemRecommendation(Dataset):
    """MovieLens PyTorch Dataset for Training
    
    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
    
    """

    def __init__(self, data, all_productIds):
        self.users, self.items, self.age, self.gender, self.location, self.category, self.price, self.brand, self.purchased = self.get_dataset(data, all_productIds)

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.age[idx], self.gender[idx], self.location[idx], self.category[idx], self.price[idx], self.brand[idx], self.purchased[idx] 

    def get_dataset(self, data, all_productIds):
        
        # Get a list of all movie IDs
        all_productIds = data['product_id'].unique()

        # Placeholders that will hold the training data
        _userid, _itemid, _age, _gender, _location, _category, _price, _brand, _purchased = [], [], [], [], [], [], [], [], []

        # This is the set of items that each user has interaction with
        user_items_dataset = set(zip(train_df['user_id'], train_df['product_id'], \
                                 train_df['age'], train_df['gender'], \
                                train_df['location'], train_df['category'], \
                                 train_df['price'], train_df['brand'], train_df['purchased']))

        user_item_set = set(zip(train_df['user_id'], train_df['product_id']))

        # 4:1 ratio of negative to positive samples
        num_negatives = 4

        for (user, item, age, gender, location, category, price, brand, purchased) in user_items_dataset:

            _userid.append(user)
            _itemid.append(item)
            _age.append(age)
            _gender.append(gender)
            _location.append(location)
            _category.append(category)
            _price.append(price)
            _brand.append(brand)
            _purchased.append(1)

            for _ in range(num_negatives):
                # randomly select an item
                negative_item = np.random.choice(all_productIds)
                # check that the user has not interacted with this item
                while (user, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_productIds)

                _userid.append(user)
                _itemid.append(negative_item)
                _age.append(age)
                _gender.append(gender)
                _location.append(location)
                _category.append(items_lookup[negative_item]['category'])
                _price.append(items_lookup[negative_item]['price'])
                _brand.append(items_lookup[negative_item]['brand'])
                _purchased.append(0)

        return torch.tensor(_userid).int(), torch.tensor(_itemid).int(), torch.tensor(_age).int(), \
            torch.tensor(_gender).int(), torch.tensor(_location).int(), \
            torch.tensor(_category).int(), torch.tensor(_price).int(), \
            torch.tensor(_brand).int(), torch.tensor(_purchased).int()


# In[5]:


import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """
    
    def __init__(self, data):
        super().__init__()
        
        self.user_embedding = nn.Embedding(num_embeddings=data['user_id'].max()+1, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=data['product_id'].max()+1, embedding_dim=8)
        self.age_embedding = nn.Embedding(num_embeddings=data['age'].max()+1, embedding_dim=8)
        self.gender_embedding = nn.Embedding(num_embeddings=data['gender'].max()+1, embedding_dim=8)
        self.location_embedding = nn.Embedding(num_embeddings=data['location'].max()+1, embedding_dim=8)
        self.category_embedding = nn.Embedding(num_embeddings=data['category'].max()+1, embedding_dim=8)
        self.price_embedding = nn.Embedding(num_embeddings=data['price'].max()+1, embedding_dim=8)
        self.brand_embedding = nn.Embedding(num_embeddings=data['brand'].max()+1, embedding_dim=8)
        
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=64)
        self.output = nn.Linear(in_features=64, out_features=1)
        self.data = data
        self.all_productIds = data['product_id'].unique()
        
    def forward(self, user_input, item_input, age_input, gender_input, location_input, category_input, price_input, brand_input):
        
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        age_embedded = self.age_embedding(age_input)
        gender_embedded = self.gender_embedding(gender_input)
        location_embedded = self.location_embedding(location_input)
        category_embedded = self.category_embedding(category_input)
        price_embedded = self.price_embedding(price_input)
        brand_embedded = self.brand_embedding(brand_input)
        
        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded, age_embedded, gender_embedded, location_embedded, category_embedded, price_embedded, brand_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        vector = nn.ReLU()(self.fc3(vector))
        vector = nn.ReLU()(self.fc4(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))
        
        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, item_input, age_input, gender_input, location_input, category_input, price_input, brand_input, purchased = batch
        predicted_labels = self(user_input, item_input, age_input, gender_input, location_input, category_input, price_input, brand_input)
        loss = nn.BCELoss()(predicted_labels, purchased.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(ItemRecommendation(self.data, self.all_productIds),batch_size=16)


# In[ ]:


model = NCF(train_df)

trainer = pl.Trainer(max_epochs=100, logger=True)

trainer.fit(model)


# In[ ]:


import pickle

# save the model
pickle.dump(model, open('ncf_model.pkl', 'wb'))

# load the model
pickled_model = pickle.load(open('ncf_model.pkl', 'rb'))


# In[ ]:


# User-item pairs for testing
test_user_item_set = set(zip(test_df['user_id'], test_df['product_id'], test_df['age'], \
                            test_df['gender'], test_df['location'], test_df['category'],test_df['price'],test_df['brand']))

# Dict of all items that are interacted with by each user
user_interacted_items = data.groupby('user_id')['product_id'].apply(list).to_dict()


hits = []
for (user, item, age, gender, location, category, price, brand) in test_user_item_set:
    interacted_items = user_interacted_items[user]
    not_interacted_items = set(all_productIds) - set(interacted_items)
    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted + [item]
    
#     print(test_df.head())
#     print(test_items)
    predicted_labels = np.squeeze(pickled_model(torch.tensor([user] * 100), 
                                        torch.tensor(test_items), 
                                        torch.tensor([age] * 100),
                                        torch.tensor([gender] * 100),
                                        torch.tensor([location] * 100),
                                        torch.tensor([items_lookup[i]['category'] for i in test_items]),
                                        torch.tensor([items_lookup[i]['price'] for i in test_items]), 
                                        torch.tensor([items_lookup[i]['brand'] for i in test_items])).detach().numpy())
    
    top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
    
    if item in top10_items:
        hits.append(1)
    else:
        hits.append(0)
        
print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))


# In[5]:


# get_ipython().system('pip install torchvision')


# In[ ]:


# what this means is that 10% of the users were recommended the actual item (among a list of 10 items)
# that they eventually interacted with


# In[9]:


import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle 

import torch
from torch.utils.data import Dataset

class ItemRecommendation(Dataset):
    """MovieLens PyTorch Dataset for Training
    
    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
    
    """

    def __init__(self, data, all_productIds):
        self.users, self.items, self.age, self.gender, self.location, self.category, self.price, self.brand, self.purchased = self.get_dataset(data, all_productIds)

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.age[idx], self.gender[idx], self.location[idx], self.category[idx], self.price[idx], self.brand[idx], self.purchased[idx] 

    def get_dataset(self, data, all_productIds):
        
        # Get a list of all movie IDs
        all_productIds = data['product_id'].unique()

        # Placeholders that will hold the training data
        _userid, _itemid, _age, _gender, _location, _category, _price, _brand, _purchased = [], [], [], [], [], [], [], [], []

        # This is the set of items that each user has interaction with
        user_items_dataset = set(zip(train_df['user_id'], train_df['product_id'], \
                                 train_df['age'], train_df['gender'], \
                                train_df['location'], train_df['category'], \
                                 train_df['price'], train_df['brand'], train_df['purchased']))

        user_item_set = set(zip(train_df['user_id'], train_df['product_id']))

        # 4:1 ratio of negative to positive samples
        num_negatives = 4

        for (user, item, age, gender, location, category, price, brand, purchased) in user_items_dataset:

            _userid.append(user)
            _itemid.append(item)
            _age.append(age)
            _gender.append(gender)
            _location.append(location)
            _category.append(category)
            _price.append(price)
            _brand.append(brand)
            _purchased.append(1)

            for _ in range(num_negatives):
                # randomly select an item
                negative_item = np.random.choice(all_productIds)
                # check that the user has not interacted with this item
                while (user, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_productIds)

                _userid.append(user)
                _itemid.append(negative_item)
                _age.append(age)
                _gender.append(gender)
                _location.append(location)
                _category.append(items_lookup[negative_item]['category'])
                _price.append(items_lookup[negative_item]['price'])
                _brand.append(items_lookup[negative_item]['brand'])
                _purchased.append(0)

        return torch.tensor(_userid).int(), torch.tensor(_itemid).int(), torch.tensor(_age).int(), \
            torch.tensor(_gender).int(), torch.tensor(_location).int(), \
            torch.tensor(_category).int(), torch.tensor(_price).int(), \
            torch.tensor(_brand).int(), torch.tensor(_purchased).int()


import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """
    
    def __init__(self, data):
        super().__init__()
        
        self.user_embedding = nn.Embedding(num_embeddings=data['user_id'].max()+1, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=data['product_id'].max()+1, embedding_dim=8)
        self.age_embedding = nn.Embedding(num_embeddings=data['age'].max()+1, embedding_dim=8)
        self.gender_embedding = nn.Embedding(num_embeddings=data['gender'].max()+1, embedding_dim=8)
        self.location_embedding = nn.Embedding(num_embeddings=data['location'].max()+1, embedding_dim=8)
        self.category_embedding = nn.Embedding(num_embeddings=data['category'].max()+1, embedding_dim=8)
        self.price_embedding = nn.Embedding(num_embeddings=data['price'].max()+1, embedding_dim=8)
        self.brand_embedding = nn.Embedding(num_embeddings=data['brand'].max()+1, embedding_dim=8)
        
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=64)
        self.output = nn.Linear(in_features=64, out_features=1)
        self.data = data
        self.all_productIds = data['product_id'].unique()
        
    def forward(self, user_input, item_input, age_input, gender_input, location_input, category_input, price_input, brand_input):
        
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        age_embedded = self.age_embedding(age_input)
        gender_embedded = self.gender_embedding(gender_input)
        location_embedded = self.location_embedding(location_input)
        category_embedded = self.category_embedding(category_input)
        price_embedded = self.price_embedding(price_input)
        brand_embedded = self.brand_embedding(brand_input)
        
        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded, age_embedded, gender_embedded, location_embedded, category_embedded, price_embedded, brand_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        vector = nn.ReLU()(self.fc3(vector))
        vector = nn.ReLU()(self.fc4(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))
        
        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, item_input, age_input, gender_input, location_input, category_input, price_input, brand_input, purchased = batch
        predicted_labels = self(user_input, item_input, age_input, gender_input, location_input, category_input, price_input, brand_input)
        loss = nn.BCELoss()(predicted_labels, purchased.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(ItemRecommendation(self.data, self.all_productIds),batch_size=16)
    
#import data
data = pd.read_csv("grocery_data2.csv")

# categorising the features into sparse/dense feature set 

# dense features are quantitative in nature
dense_features = ['quantity', 'age', 'price',]

#sparse features are categorical in nature
sparse_features = ['user_id', 'product_id', 'gender', 'location', 'product_name', 'category',  'brand', 'day_of_week', 'time_of_day']

# encoding function
def encoding(data,feat,encoder):
    data[feat] = encoder.fit_transform(data[feat])
# encoding for categorical features
[encoding(data,feat,LabelEncoder()) for feat in sparse_features]

# data imputation for missing values
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)

data['price'] = (data['price'] * 1000 ).astype(int)

items_lookup = {}
users_lookup = {}

for i in range(len(data)):
    items_lookup[data['product_id'].iloc[i]] = data.iloc[i][['category','price','brand']]
    users_lookup[data['user_id'].iloc[i]] = data.iloc[i][['age','location','gender']]

def predict(user_id, top_num, include_item_bought_before = True):
    
    # load the model
    saved_model = pickle.load(open('ncf_model.pkl', 'rb'))

    # Dict of all items that are interacted with by each user
    user_interacted_items = data.groupby('user_id')['product_id'].apply(list).to_dict()

    # Get a list of all product IDs
    all_productIds = data['product_id'].unique()
    
    interacted_items = user_interacted_items[user_id]
    not_interacted_items = set(all_productIds) - set(interacted_items)
    
    test_items = list(set(all_productIds)) if include_item_bought_before else list(set(not_interacted_items))
    
    predicted_labels = np.squeeze(saved_model(torch.tensor([user_id] * len(test_items)), 
                                        torch.tensor(test_items), 
                                        torch.tensor([users_lookup[user_id]['age']] * len(test_items)),
                                        torch.tensor([users_lookup[user_id]['gender']] * len(test_items)),
                                        torch.tensor([users_lookup[user_id]['location']] * len(test_items)),
                                        torch.tensor([items_lookup[i]['category'] for i in test_items]),
                                        torch.tensor([items_lookup[i]['price'] for i in test_items]), 
                                        torch.tensor([items_lookup[i]['brand'] for i in test_items])).detach().numpy())

    top_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:top_num].tolist()]

    return top_items
    
predict(2,10,False)


# In[ ]:


# !pip install pytorch-lightning
import pickle
pickle.dump('',open('Neural_collberative.pkl','wb'))


# In[ ]:


# https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e

