store = [
    {
        "text": "How are you?",
        "embeddings": [1.0, 2.0, 3.0]
    },
    {
        "text": "Where are you?",
        "embeddings": [1.0, 2.0, 3.0]
    }
]

store_embeddings = [item["embeddings"] for item in store]
print(store_embeddings)