import os

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import warnings
warnings.filterwarnings("ignore")

EMBEDDINGS_DIR = './embeddings'

def load_articles(file_path):
    with open(file_path, 'r') as f:
        content = f.readlines()
    return content

def get_rag(article_name):
    
    print(f'\n[DEBUG] : Loading embeddings for {article_name} ...\n')

    embedding_fpath = os.path.join(EMBEDDINGS_DIR, article_name)
    if(os.path.exists(embedding_fpath)) :
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.load_local(folder_path=embedding_fpath, embeddings=embeddings)
        return db
    else :
        print(f'\n[DEBUG] : Embeddings not found at : {article_name} ...\n')
        return None
