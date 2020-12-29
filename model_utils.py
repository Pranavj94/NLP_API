import numpy as np
import configs
import text_utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class BiLSTM(nn.Module):
    def __init__(self,hidden_size,num_classes,dropout,max_features,embed_size,embedding_matrix):
        super(BiLSTM, self).__init__()
        #self.hidden_size = hidden_size
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*4 , 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, num_classes)


    def forward(self, x):
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out
    

def predict_single(x,model,tokenizer,le):    
    # clean the text
    processor=text_utils.text_processor()
    x = processor.process_text(x)
    # tokenize
    x = tokenizer.texts_to_sequences([x])
    # pad
    x = pad_sequences(x, maxlen=configs.maxlen)
    # create dataset
    x = torch.tensor(x, dtype=torch.long)

    out = model(x).detach()
    soft_out = F.softmax(out).cpu().numpy()

    pred = soft_out.argmax(axis=1)
    prob = np.amax(soft_out)

    pred = le.classes_[pred]
    return (pred[0],round(float(prob),2))

