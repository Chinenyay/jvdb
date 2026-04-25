A learning project where I am implementing a vector database from scratch

Level 0 [in progress]: use python lists to store corresponding texts and vector embeddings, and retrieve most semantically similar items given a query. use brute force search - comparing the query vector with every single pair in store and return top k matching items in store
leve 0.5: use numpy arrays to store the text-embeddings pairs

todo on level 0
db methods to add
    - upsert()
    - save_to_disk()
    - filter()
- add tests
- add profiling
- use dataclasses to write type annotations for function args, to make code more readable?

- add a filter step in search
- measure search speed
- measure memory usage
- profile and compare python list and numpy array implementations


