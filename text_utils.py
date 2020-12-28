import torch
import numpy as np
import re
import random
import os


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

class text_processor:
    def __init__(self):
        pass
    def identify_weblinks(self,text):
        return(re.sub(r'http\S+', 'weblink', text))
    def identify_tags(self,text):
        return(re.sub(r'@\S+', 'tag', text))
    def remove_punctuations(self,text):
        return(re.sub(r'[^\w\s]', '', text))
    def lower_case(self,text):
        return(text.lower())
    def process_text(self,text):
        text=self.identify_tags(text)
        text=self.identify_weblinks(text)
        text=self.remove_punctuations(text)
        text=self.lower_case(text)
        return(text)

# Loading the embedding matrix from glove pretrained embedding

def load_glove(word_index,max_features):
    EMBEDDING_FILE = './glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding="utf8"))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix
