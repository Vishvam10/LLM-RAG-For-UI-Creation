import os

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import warnings
warnings.filterwarnings("ignore")

EMBEDDINGS_DIR = './embeddings'

def get_vector_embeddings(article_name):
    
    print(f'\n[DEBUG] : Loading embeddings for {article_name} ...\n')

    embedding_fpath = os.path.join(EMBEDDINGS_DIR, article_name)
    if(os.path.exists(embedding_fpath)) :
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.load_local(folder_path=embedding_fpath, embeddings=embeddings)
        return db
    else :
        print(f'\n[DEBUG] : Embeddings not found at : {article_name} ...\n')
    return None

def create_rag(article_names, cache_name) :
    
    rag = get_vector_embeddings(article_names[0])

    for article in article_names[1:] :
        db = get_vector_embeddings(article)
        rag.merge_from(db)
        print(f'\n[DEBUG] : Merged embeddings : {article} -> {len(rag.docstore._dict)} ...\n')

    print('[DEBUG] : Caching RAG ...\n')

    if(cache_name != "") :

        embedding_fpath = os.path.join(EMBEDDINGS_DIR, cache_name)
        db.save_local(embedding_fpath)
    
    return rag
    

# if __name__ == '__main__' :

#     db1 = get_rag('bootstrap_docs')
#     db2 = get_rag('bootstrap_examples') 
#     db3 = get_rag('bootstrap_code')
    
#     print(len(db1.docstore._dict))
    
#     db1.merge_from(db2)
#     print(len(db1.docstore._dict))
    
#     db1.merge_from(db3)
#     print(len(db1.docstore._dict))

