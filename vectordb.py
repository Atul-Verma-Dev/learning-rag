import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_client.delete_collection("my_docs")
