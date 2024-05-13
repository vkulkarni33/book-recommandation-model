import numpy as np
import pandas as pd
import torch as torch
import tensorflow as tf
from  torch_geometric.data  import  Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset with 'title', 'author', 'genre', and 'user_rating' columns
books_df = pd.read_csv(r'/books.csv')
titles = books_df['title'].values
authors = books_df['author'].values
genres = books_df['genre'].values
user_ratings = books_df['user_rating'].values

# Preprocess features
title_encoder = LabelEncoder()
author_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

titles_encoded = title_encoder.fit_transform(titles)
authors_encoded = author_encoder.fit_transform(authors)
genres_encoded = genre_encoder.fit_transform(genres)

# Normalize user ratings
scaler = MinMaxScaler()
user_ratings_normalized = scaler.fit_transform(user_ratings.reshape(-1, 1)).squeeze()

# Create node features by concatenating encoded features and normalized ratings
node_features = torch.tensor(
    list(zip(titles_encoded, authors_encoded, genres_encoded, user_ratings_normalized)),
    dtype=torch.float
)

# Create a graph (dummy example)
edge_index = torch.tensor([[i, i+1] for i in range(len(titles)-1)], dtype=torch.long).t().contiguous()

# Create graph data
data = Data(x=node_features, edge_index=edge_index)

# Define a GNN model that handles additional features
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, len(titles))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return x

# Instantiate the model
model = GNN()

# Define a function for recommendations that considers additional features
def recommend(book_title, model, data, top_k=5):
    # Encode the title
    title_encoded = title_encoder.transform([book_title])[0]
    
    # Get the prediction scores from the model
    model.eval()
    with torch.no_grad():
        out = model(data)
    
    # Get top k recommendations
    scores = out[title_encoded]
    _, indices = torch.topk(scores, top_k)
    recommendations = title_encoder.inverse_transform(indices.cpu().numpy())
    
    return recommendations

print("Code runs successfully")

#Spilliting data
from sklearn.model_selection import train_test_split

# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(books_df, test_size=0.2, random_state=42)

# Now you can preprocess the features and create the graph data for each set separately
print("Data split")



# Preprocess features and create graph data for training set
titles_train = train_df['title'].values
authors_train = train_df['author'].values
genres_train = train_df['genre'].values
user_ratings_train = train_df['user_rating'].values

titles_encoded_train = title_encoder.transform(titles_train)
authors_encoded_train = author_encoder.transform(authors_train)
genres_encoded_train = genre_encoder.transform(genres_train)
user_ratings_normalized_train = scaler.transform(user_ratings_train.reshape(-1, 1)).squeeze()

node_features_train = torch.tensor(
    list(zip(titles_encoded_train, authors_encoded_train, genres_encoded_train, user_ratings_normalized_train)),
    dtype=torch.float
)

edge_index_train = torch.tensor([[i, i+1] for i in range(len(titles_train)-1)], dtype=torch.long).t().contiguous()

data_train = Data(x=node_features_train, edge_index=edge_index_train)
data_train.y = torch.tensor(user_ratings_train, dtype=torch.float)



# Preprocess features and create graph data for testing set
titles_test = test_df['title'].values
authors_test = test_df['author'].values
genres_test = test_df['genre'].values
user_ratings_test = test_df['user_rating'].values

titles_encoded_test = title_encoder.transform(titles_test)
authors_encoded_test = author_encoder.transform(authors_test)
genres_encoded_test = genre_encoder.transform(genres_test)
user_ratings_normalized_test = scaler.transform(user_ratings_test.reshape(-1, 1)).squeeze()

node_features_test = torch.tensor(
    list(zip(titles_encoded_test, authors_encoded_test, genres_encoded_test, user_ratings_normalized_test)),
    dtype=torch.float
)

edge_index_test = torch.tensor([[i, i+1] for i in range(len(titles_test)-1)], dtype=torch.long).t().contiguous()

data_test = Data(x=node_features_test, edge_index=edge_index_test)
data_test.y = torch.tensor(user_ratings_test, dtype=torch.float)



#Training
# Import necessary PyTorch packages
import torch.optim as optim
from torch.nn import MSELoss

# Define a loss function
loss_fn = MSELoss()

# Define an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

#define a training loop
def train(model, data, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear gradients
        out = model(data)  # Forward pass
        
        # Flatten the output and target tensors
        out_flat = out.view(-1)
        target_flat = data.y.view(-1)
        
        # Print tensor shapes for debugging
        print(f"Out shape: {out_flat.shape}, Target shape: {target_flat.shape}")
        
        # Compute loss for all nodes
        loss = loss_fn(out_flat, target_flat)
        
        # Apply mask to the flattened tensors
        out_masked = out_flat[data.train_mask.view(-1)]
        target_masked = target_flat[data.train_mask.view(-1)]
        
        # Compute the masked loss
        masked_loss = loss_fn(out_masked, target_masked)
        
        masked_loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        print(f"Epoch: {epoch+1}, Loss: {masked_loss.item()}")

print("Model trained")
