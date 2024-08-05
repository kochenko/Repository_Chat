#!/usr/bin/env python3
import os
import glob
import time
import logging
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from constants import CHROMA_SETTINGS

# Configurar logs
logging.basicConfig(filename='ingest.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Load environment variables
persist_directory = 'db_orion'
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
chunk_size = 500
chunk_overlap = 50

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".py": (TextLoader, {"encoding": "utf8"}),
    ".sh": (TextLoader, {"encoding": "utf8"}),
    ".yaml": (TextLoader, {"encoding": "utf8"}),
    ".yml": (TextLoader, {"encoding": "utf8"})
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        try:
            return loader.load()
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            logging.error(f"Failed to load {file_path}: {e}")
            return []

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    num_processes = 1  # Use only 1 process to reduce CPU load
    batch_size = 5  # Number of documents to process in each batch

    documents = []
    with Pool(processes=num_processes) as pool:
        for i in range(0, len(filtered_files), batch_size):
            batch_files = filtered_files[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1} with files: {batch_files}")
            results = []
            with tqdm(total=len(batch_files), desc=f'Loading batch {i//batch_size + 1}', ncols=80) as pbar:
                for docs in pool.imap_unordered(load_single_document, batch_files):
                    results.extend(docs)
                    pbar.update()
                    time.sleep(0.5)  # Add a small delay to reduce CPU load
            documents.extend(results)

    return documents

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    logging.info(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    logging.info(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            if len(list_index_files) > 3:
                return True
    return False

def create_embeddings_in_batches(db, texts, batch_size=5000):
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if batch:
            db.add_texts([t.page_content for t in batch], metadatas=[t.metadata for t in batch])
        print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        logging.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        time.sleep(120)  # Add a longer delay to reduce CPU load
        with open('embedding_progress.log', 'a') as f:
            f.write(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}\n")

def main():
    # Forçar uso da CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        logging.info(f"Creating embeddings. May take some minutes...")
        create_embeddings_in_batches(db, texts, batch_size=5000)  # Create embeddings in batches
    else:
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        logging.info(f"Creating embeddings. May take some minutes...")
        if texts:  # Ensure that texts is not empty
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
            create_embeddings_in_batches(db, texts, batch_size=5000)  # Create embeddings in batches
        else:
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
    logging.info("Ingestion complete! You can now run privateGPT.py to query your documents")

if __name__ == "__main__":
    main()
