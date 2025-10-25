# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-0.6B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = ["What is TiDB",
           "Please introduce TiDB",
           "What is TiDB used for",
           "How good is TiDB",
           "Does TiDB support FOREIGN KEY"]

answer = "What is TiDB"

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode([answer])

# Compute the (cosine) similarity between the query and document embeddings
for i, query_embedding in enumerate(query_embeddings):
    similarities = model.similarity(query_embedding, document_embeddings[0])
    print(f"Similarities for query: {i}")
    print(similarities)
    # tensor([0.7646])  # Example output