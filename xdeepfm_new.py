import sys
# importing libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import xDeepFM
from deepctr.feature_column  import  SparseFeat, DenseFeat,get_feature_names
import numpy as np
import pickle

data = pd.read_csv('./grocery_data2.csv')

data['purchased'] = 1
# categorising the features into sparse/dense feature set 

# dense features are quantitative in nature
dense_features = ['age', 'price',]

#sparse features are categorical in nature
sparse_features = ['user_id', 'product_id', 'gender', 'location', 'category',  'brand']

# data imputation for missing values
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)
# creating target variable
target = ['purchased']

# encoding function
def encoding(data,feat,encoder):
    data[feat] = encoder.fit_transform(data[feat])
# encoding for categorical features
[encoding(data,feat,LabelEncoder()) for feat in sparse_features]
# Using normalization for dense feature
mms = MinMaxScaler(feature_range=(0,1))
data[dense_features] = mms.fit_transform(data[dense_features])
# creating a 4 bit embedding for every sparse feature
sparse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4) \
for i,feat in enumerate(sparse_features)]

# creating a dense feat
dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

# features to be used for dnn part of xdeepfm
dnn_feature_columns = sparse_feature_columns + dense_feature_columns
# features to be used for linear part of xdeepfm
linear_feature_columns = sparse_feature_columns + dense_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

items_lookup = {}
users_lookup = {}

for i in range(len(data)):
    items_lookup[data['product_id'].iloc[i]] = data.iloc[i][['category','price','brand']]
    users_lookup[data['user_id'].iloc[i]] = data.iloc[i][['age','location','gender']]

# Get a list of all movie IDs
all_productIds = data['product_id'].unique()

user_item_set = set(zip(data['user_id'], data['product_id']))

# This is the set of items that each user has interaction with
user_items_dataset = set(zip(data['user_id'], data['product_id'], \
                         data['age'], data['gender'], \
                        data['location'], data['category'], \
                         data['price'], data['brand'], data['purchased']))

def predict(user_id, top_num,include_item_bought_before=True):
    
    fpkl= open('xdeepfm_model.pkl', 'rb')    #Python 3     
    weights = pickle.load(fpkl)
    fpkl.close()

    model = xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(512, 512),\
        cin_layer_size=(512, 512), \
        cin_split_half=True, cin_activation='relu'\
        ,l2_reg_linear=1e-05,\
        l2_reg_embedding=1e-05, l2_reg_dnn=0, l2_reg_cin=0, \
        # init_std=0.0001,\
        seed=1024, dnn_dropout=0.2,dnn_activation='relu', \
        dnn_use_bn=True, task='binary')
    model.set_weights(weights)
    
    product_array, location_array, gender_array, age_array, category_array, price_array, brand_array = [],[],[],[],[],[],[]
    
    for item in data['product_id'].unique():
        if (user_id, item) in user_item_set and not(include_item_bought_before):
            continue
        product_array.append(item)
        location_array.append(users_lookup[user_id]['location'])
        gender_array.append(users_lookup[user_id]['gender'])
        age_array.append(users_lookup[user_id]['age'])
        category_array.append(items_lookup[item]['category'])
        price_array.append(items_lookup[item]['price'])
        brand_array.append(items_lookup[item]['brand'])
        
    model_input = {'user_id':np.array([user_id] * len(age_array)).astype(int),
                    'product_id': np.array(product_array).astype(int), 
                    'gender':np.array(gender_array).astype(int),
                    'location':np.array(location_array).astype(int),
                    'category':np.array(category_array).astype(int),
                    'brand':np.array(brand_array).astype(int),
                    'age':np.array(age_array).astype(float),
                    'price':np.array(price_array).astype(int),}
    
    predictions = np.reshape(model.predict(model_input),(-1))
    
    return [data['product_id'].unique()[i] for i in np.argsort(predictions)[::-1][0:top_num].tolist()] 

def main(argv):
    print('Xdeep FM Args : ',argv)
    userid = int(argv[0])
    print('X deep FM Predicition : ',predict(userid, 10, False))
    print('X deep FM Predicition Type: ',type(predict(userid, 10, False)))
    return predict(userid, 10, False)
    # args is a list of the command line args

if __name__ == "__main__":
   main(sys.argv[1:])
