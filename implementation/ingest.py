import os
import numpy as np
import glob
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# home 
home = os.path.expanduser("~")
# ingest file location
here = Path(__file__).resolve().parent
# project path 
proj_path = Path(__file__).resolve().parents[1]

# path to knowledge base
knowledge_base =  os.path.join(proj_path, "knowledge-base")

# load config
import yaml
with open(os.path.join(proj_path, "config.yml"), "r") as f:
    raw_yml = f.read()
config = yaml.safe_load(raw_yml)

# vector db name
db_name= config["DB_NAME"]
db_path = os.path.join(proj_path, db_name)
# embedding
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def fetch_documents():
    
    documents = []
    # folder/document paths in knowledge base
    folder_path = glob.glob(f"{knowledge_base}/*")
    
    for folder in folder_path:
        # get document folder name
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(path=folder,
                                 glob="**/*.md", 
                                 loader_cls=TextLoader, 
                                 loader_kwargs={"encoding":"utf-8"})
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents
    
def create_chunks(documents):
    # Divide into chunks using RecursiveCharacterTextSpliter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=200)
    # we pass in the documents created using langchain `DirectoryLoader`
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    if os.path.exists(db_path):
        Chroma(persist_directory=db_name,
              embedding_function=embedding_function).delete_collection()

    vectorstore = Chroma.from_documents(documents=chunks,
                         persist_directory=db_name,
                         embedding=embedding_function)
    
    # collection = vectorstore._collection
    # count = collection.count()

    return vectorstore


if __name__ == "__main__":
    docunments = fetch_documents()
    chunks = create_chunks(docunments)
    create_embeddings(chunks)
    print("Ingestion Complete")

