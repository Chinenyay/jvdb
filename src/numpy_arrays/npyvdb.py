import numpy as np
from openai import OpenAI
import os
from pathlib import Path
import json

'''
    methods:
    __init__(self, name, data)
    
    generate_embeddings_for_stored_data()
    normalize_stored_embeddings()
    normalize_query_embeddings()
    save_to_disk()
    load_from_disk()
    search()
    '''


class NPVectorDB:
    def __init__(self, name: str, data: list[str]):
        self.name = name
        self.data = data
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "text-embedding-3-large"
        self.embeddings = self._create_embeddings_array()
    
    def _generate_embeddings(self, input: str):
        response = self.client.embeddings.create(
            input=input,
            model=self.model
        )
        return response.data[0].embedding
    
    def _generate_embeddings_for_stored_data(self, data):
        data_list = []
        for item in data:
            emb = self._generate_embeddings(item)
            data_list.append(emb)
        return data_list

    def _create_embeddings_array(self):
        return np.array(self._generate_embeddings_for_stored_data(self.data), dtype=np.float32)
    
    def _generate_query_embeddings(self, query):
        response = self.client.embeddings.create(
            input=query,
            model=self.model
        )
        return response.data[0].embedding

    def _norm(self, embeddings, axis: int):
        norms = np.linalg.vector_norm(embeddings, axis=axis, keepdims=True)
        return norms
    
    def _normalize(self, embeddings, norms):
        normalized = embeddings / norms
        return normalized
    
    def _normalize_stored_embeddings(self):
        embs = self.embeddings
        emb_norms = self._norm(embs, axis=1)
        normalized_embs = self._normalize(embs, emb_norms)
        self.embeddings = normalized_embs
        return self.embeddings
    
    def _normalize_query_embeddings(self, query):
        query_emb = self._generate_query_embeddings(query)
        query_norm = self._norm(query_emb, axis=0)
        normalized_query_emb = self._normalize(query_emb, query_norm)
        return normalized_query_emb
    
    def _cosine_similarity(self, vector_embs, query_emb):
        result = vector_embs @ query_emb
        return result
    
    def search(self, query, k: int):
        vector_embs = self._normalize_stored_embeddings()
        query_emb = self._normalize_query_embeddings(query)
        similarity_scores = self._cosine_similarity(vector_embs, query_emb)
        sorted_scores_indices = np.argsort(similarity_scores)[::-1][:k]

        top_k_matches = []
        for index in sorted_scores_indices:
            top_k_matches.append({
                "text": self.data[index],
                "score": float(similarity_scores[index])
            })
        return top_k_matches

    def load_from_disk(self):
        filename = f"src/datastore/{self.name}.json"
        path = Path(filename)
        if path.exists() == False:
            return FileNotFoundError("File does not exist. Create file first.")
        stored_embeddings = json.loads(path)



def main():
    test_data = ["Paris, France", "London, UK", "Hello world"]
    np_db = NPVectorDB(name="test_db", data=test_data)
    norm_embs = np_db._normalize_stored_embeddings()
    # print(norm_embs)
    test_query = "programming"
    normed_query_emb = np_db._normalize_query_embeddings(test_query)

    # cosine_sim = np_db._cosine_similarity(norm_embs, normed_query_emb)
    print(np_db.search(test_query, 2))
    
    # print(cosine_sim)
if __name__ == "__main__":
    main()