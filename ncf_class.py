import pickle
from NCF_Model import ItemRecommendation, NCF
from django.http.response import JsonResponse, HttpResponse
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle 
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pytorch_lightning as pl
# from torch.utils.data import DataLoaderx
from torch.utils.data import DataLoader

# encoding function
def encoding(data,feat,encoder):
    data[feat] = encoder.fit_transform(data[feat])

def predict(user_id, top_num, include_item_bought_before = True):

    #import data
    data = pd.read_csv("grocery_data2.csv")

    # categorising the features into sparse/dense feature set 

    # dense features are quantitative in nature
    dense_features = ['quantity', 'age', 'price',]

    #sparse features are categorical in nature
    sparse_features = ['user_id', 'product_id', 'gender', 'location', 'product_name', 'category',  'brand', 'day_of_week', 'time_of_day']

    
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
        
    print('Load NCF.')
    saved_model = NCF(data)

    print('Goes to Predict.')    
    # load the model
    saved_model = pickle.load(open('ncf_model.pkl', 'rb'))
    print('Model Loaded')
    # Dict of all items that are interacted with by each user
    user_interacted_items = data.groupby('user_id')['product_id'].apply(list).to_dict()

    # Get a list of all product IDs
    all_productIds = data['product_id'].unique()
    
    interacted_items = user_interacted_items[user_id]
    not_interacted_items = set(all_productIds) - set(interacted_items)
    
    test_items = list(set(all_productIds)) if include_item_bought_before else list(set(not_interacted_items))
    print('Items Categorize')
    predicted_labels = np.squeeze(saved_model(torch.tensor([user_id] * len(test_items)), torch.tensor(test_items), torch.tensor([users_lookup[user_id]['age']] * len(test_items)),torch.tensor([users_lookup[user_id]['gender']] * len(test_items)),torch.tensor([users_lookup[user_id]['location']] * len(test_items)),torch.tensor([items_lookup[i]['category'] for i in test_items]),torch.tensor([items_lookup[i]['price'] for i in test_items]), torch.tensor([items_lookup[i]['brand'] for i in test_items])).detach().numpy())
    print('Data Sorted As oer Our Need')
    top_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:top_num].tolist()]
    print('Converted in To List')
    return top_items