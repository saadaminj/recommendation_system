import pickle
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