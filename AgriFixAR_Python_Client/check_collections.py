# check_collections.py

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

for name in ["agrifix", "langchain"]:
    db = Chroma(
        collection_name=name,
        persist_directory="./chroma_db",
        embedding_function=emb,
    )

    print(name, "count =", db._collection.count())