import numpy as np
from openai import OpenAI
import os
from pathlib import Path
import json
from typing import Optional, Any

class NPVectorDB:
    def __init__(self, name: str):
        self.name = name
        self.data_file_name = f"src/datastore/{self.name}.data.json"
        self.emb_file_name = f"src/datastore/{self.name}.emb.json"
        self.data: list[str] = self.load_from_disk(self.data_file_name)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "text-embedding-3-large"
        self.embeddings = self.load_from_disk(self.emb_file_name)


    def add_data(self, new_data: list[str]):
        self.data.extend(new_data)
    
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

    def load_from_disk(self, filename) -> list[str]:
        filename = filename
        path = Path(filename) 
        if path.exists() == False:
            return []
        with open(path, "r+") as f:
            data = json.load(f)
        return data
    

#todo: save embeddings after normalization to npz files.
    def save_to_disk(self):
        data_file_name = self.data_file_name
        emb_file_name = self.emb_file_name

        data_path = Path(data_file_name)
        emb_path = Path(emb_file_name)

        with open(data_path, "w+") as f:
            json.dump(self.data, f)
        
        with open(emb_path, "w+") as f:
            json.dump(self._create_embeddings_array().tolist(), f)
        
        return {"message": f"saved data and embeddings to {data_path}, {emb_path} respectively."}

def main():
    # test_data = ["Paris, France", "London, UK", "Hello world"]

    # test_data_2 = ["When are you home?", "I love New York"]
    # test_data_3 = ["Hi"]
    np_db = NPVectorDB(name="test_db")

    # np_db.add_data(test_data)
    # np_db.add_data(test_data_2)
    # np_db._generate_embeddings_for_stored_data(test_data_3)
    # np_db.add_data(test_data_3)

    # norm_embs = np_db._normalize_stored_embeddings()
    # print(norm_embs)
    # test_query = "programming"
    # normed_query_emb = np_db._normalize_query_embeddings(test_query)

    # cosine_sim = np_db._cosine_similarity(norm_embs, normed_query_emb)
    # print(np_db.search(test_query, 2))
    # print(np_db.save_to_disk())
    # print(np_db.data)
    # print(np_db.embeddings)
    # print(np_db.data_file_name)
    
    # print(cosine_sim)
if __name__ == "__main__":
    main()