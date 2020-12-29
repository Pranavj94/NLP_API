from fastapi import FastAPI
import torch
import model_utils
import configs
from pydantic import BaseModel
import numpy as np
import pickle
import text_utils

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="NLP tester", description="API for NLP use cases", version="1.0")

# Loading the model
model = model_utils.BiLSTM(configs.hidden_size,configs.num_classes,configs.dropout,
configs.max_features,configs.embed_size,np.zeros((configs.max_features, configs.embed_size)))
model.load_state_dict(torch.load('./model_files/bilstm.pt'))

# Loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
# Loading label encoder
with open('label_encoder.pkl', 'rb') as handle:
    le = pickle.load(handle)

    
#################################################################################    

def predict_single(x):    
    # clean the text
    processor=text_utils.text_processor()
    x = processor.process_text(x)
    # tokenize
    x = tokenizer.texts_to_sequences([x])
    # pad
    x = pad_sequences(x, maxlen=configs.maxlen)
    # create dataset
    x = torch.tensor(x, dtype=torch.long)

    pred = model(x).detach()
    pred = F.softmax(pred).cpu().numpy()

    pred = pred.argmax(axis=1)

    pred = le.classes_[pred]
    return pred[0]

    
# define the Input class
class Input(BaseModel):
    text : str

        
@app.get('/')
async def checker():
    return('Hello, welcome to NLP API')



@app.put('/predict')
async def get_prediction(d:Input):
    data = d.text
    prediction = predict_single(data).tolist()
    return {"prediction": prediction}



