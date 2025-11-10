import torch
import torch.nn as nn


class SimpleGNN(nn.Module):
    def __init__(self, n_users, n_merchants, embedding_dim=64):
        super(SimpleGNN, self).__init__()
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.merchant_embedding = nn.Embedding(n_merchants, embedding_dim)
        
        self.demo_fc = nn.Linear(2, 16)
        
        self.gc1 = nn.Linear(embedding_dim * 2 + 16, 128)
        self.gc2 = nn.Linear(128, 64)
        self.gc3 = nn.Linear(64, 32)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        
        self.output = nn.Linear(32, 2)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.merchant_embedding.weight)
    
    def forward(self, user_ids, merchant_ids, age_range, gender):
        user_emb = self.user_embedding(user_ids)
        merchant_emb = self.merchant_embedding(merchant_ids)
        
        demo = torch.stack([age_range, gender], dim=1)
        demo_emb = torch.relu(self.demo_fc(demo))
        
        x = torch.cat([user_emb, merchant_emb, demo_emb], dim=1)
        
        x = self.gc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.gc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.gc3(x)
        x = torch.relu(x)
        
        x = self.output(x)
        return x


class EmbeddingModel(nn.Module):
    def __init__(self, n_users, n_merchants, embedding_dim=50):
        super(EmbeddingModel, self).__init__()
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.merchant_embedding = nn.Embedding(n_merchants, embedding_dim)
        
        self.demo_fc = nn.Linear(2, 16)
        
        self.fc1 = nn.Linear(embedding_dim * 2 + 16, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.output = nn.Linear(32, 2)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.merchant_embedding.weight)
    
    def forward(self, user_ids, merchant_ids, age_range, gender):
        user_emb = self.user_embedding(user_ids)
        merchant_emb = self.merchant_embedding(merchant_ids)
        
        demo = torch.stack([age_range, gender], dim=1)
        demo_emb = torch.relu(self.demo_fc(demo))
        
        x = torch.cat([user_emb, merchant_emb, demo_emb], dim=1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        return x


class DeepFM(nn.Module):
    def __init__(self, n_users, n_merchants, embedding_dim=32):
        super(DeepFM, self).__init__()
        
        self.user_embedding_fm = nn.Embedding(n_users, embedding_dim)
        self.merchant_embedding_fm = nn.Embedding(n_merchants, embedding_dim)
        
        self.user_embedding_dnn = nn.Embedding(n_users, embedding_dim)
        self.merchant_embedding_dnn = nn.Embedding(n_merchants, embedding_dim)
        
        self.user_bias = nn.Embedding(n_users, 1)
        self.merchant_bias = nn.Embedding(n_merchants, 1)
        
        self.demo_fc = nn.Linear(2, 16)
        
        dnn_input_dim = embedding_dim * 2 + 16
        self.dnn = nn.Sequential(
            nn.Linear(dnn_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.output = nn.Linear(32, 2)
        
        nn.init.xavier_uniform_(self.user_embedding_fm.weight)
        nn.init.xavier_uniform_(self.merchant_embedding_fm.weight)
        nn.init.xavier_uniform_(self.user_embedding_dnn.weight)
        nn.init.xavier_uniform_(self.merchant_embedding_dnn.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.merchant_bias.weight)
    
    def forward(self, user_ids, merchant_ids, age_range, gender):
        user_emb_fm = self.user_embedding_fm(user_ids)
        merchant_emb_fm = self.merchant_embedding_fm(merchant_ids)
        
        fm_interaction = torch.sum(user_emb_fm * merchant_emb_fm, dim=1, keepdim=True)
        linear = self.user_bias(user_ids) + self.merchant_bias(merchant_ids)
        
        user_emb_dnn = self.user_embedding_dnn(user_ids)
        merchant_emb_dnn = self.merchant_embedding_dnn(merchant_ids)
        
        demo = torch.stack([age_range, gender], dim=1)
        demo_emb = torch.relu(self.demo_fc(demo))
        
        dnn_input = torch.cat([user_emb_dnn, merchant_emb_dnn, demo_emb], dim=1)
        dnn_output = self.dnn(dnn_input)
        
        x = self.output(dnn_output)
        return x
