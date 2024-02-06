import os
import nest_asyncio
import concurrent.futures
from multiprocessing import Pool, cpu_count

from langchain_core.documents.base import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import warnings
warnings.filterwarnings("ignore")

nest_asyncio.apply()

DATA_DIR = './data'
EMBEDDINGS_DIR = './embeddings'
CACHE_DIR = './temp'

def load_articles(file_path):
    with open(file_path, 'r') as f:
        content = f.readlines()
    return content

def create_vector_embeddings(article_name, article_file):
    print(f'\n[DEBUG] : Starting for {article_file} ...\n')

    # Articles to index
    fpath = os.path.join(DATA_DIR, article_file)
    articles = load_articles(fpath)

    print('[DEBUG] : Scraping data ...\n')

    # Scrapes the above articles
    loader = AsyncChromiumLoader(articles)
    docs = loader.load()

    print('[DEBUG] : Transforming data ...\n')

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    print('[DEBUG] : Chunking data ...\n')

    # Chunk text
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(docs_transformed)

    print('[DEBUG] : Creating embeddings ...\n')

    # Load chunked documents into the FAISS index
    db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    print('[DEBUG] : Caching embeddings ...\n')

    embedding_fpath = os.path.join(EMBEDDINGS_DIR, article_name)
    db.save_local(embedding_fpath)


def process_article(article):
    article_name, article_file = article
    return create_vector_embeddings(article_name, article_file)

if __name__ == '__main__':
    aname = input('article names : ').split(' ')
    apath = [f'{x}.txt' for x in aname]
    articles = list(zip(aname, apath))

    # Use multithreading to speed up loading, scraping, and transforming
    max_workers = cpu_count() - 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_article, articles)

    # bootstrap_docs bootstrap_examples bootstrap_code