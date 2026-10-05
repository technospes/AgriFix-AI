from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3', cache_folder='./bge_cache')
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")