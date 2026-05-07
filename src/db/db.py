import numpy as np
from openai import OpenAI
import os
from pathlib import Path
import json

from dotenv import load_dotenv

load_dotenv()

class VectorDB:
    def __init__(self, name: str):
        self.name = name
        self.data_file_name = f"src/datastore/{self.name}.data.json"
        self.emb_file_name = f"src/datastore/{self.name}.emb.npz"
        self.data: list[str] = self.load_from_disk(self.data_file_name)
        self.embeddings = self.load_from_disk(self.emb_file_name)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "text-embedding-3-large"
        # self.embeddings = self._normalize_stored_embeddings(self.load_from_disk(self.emb_file_name), axis=1)

    def upsert(self, new_data: list[str]):
        new_data_embeddings = np.array(self._generate_embeddings_for_stored_data(new_data), dtype=np.float32)
        new_data_embeddings = self._normalize(new_data_embeddings, self._norm(new_data_embeddings, axis=1))

        if len(self.embeddings) == 0:
            self.embeddings = new_data_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_data_embeddings])
            
        self.data.extend(new_data)
        self.save_to_disk()

    
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
    
    def _normalize_stored_embeddings(self, embeddings, axis):
        embs = embeddings
        emb_norms = self._norm(embs, axis=axis)
        normalized_embs = self._normalize(embs, emb_norms)
        return normalized_embs
    
    def _normalize_query_embeddings(self, query):
        query_emb = self._generate_query_embeddings(query)
        query_norm = self._norm(query_emb, axis=0)
        normalized_query_emb = self._normalize(query_emb, query_norm)
        return normalized_query_emb
    
    def _cosine_similarity(self, vector_embs, query_emb) -> list[float]:
        result = vector_embs @ query_emb
        return result
    
    def search(self, query, k: int):
        if k < 1:
            raise ValueError("number of matches cannot be less than 1.")
        if k > len(self.embeddings) or k < len(self.data):
            raise ValueError("number of matches cannot be more than number of records.")
        vector_embs = self.embeddings
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
        file_name = filename
        path = Path(file_name) 
        if path.exists() == False:
            return []
        if ".npz" in path.name:
            data = np.load(path)
            return data["arr_0"]
        
        with open(path, "r+") as f:
            data = json.load(f)
        return data

    def save_to_disk(self):
        data_file_name = self.data_file_name
        emb_file_name = self.emb_file_name

        data_path = Path(data_file_name)
        emb_path = Path(emb_file_name)

        with open(data_path, "w+") as f:
            json.dump(self.data, f)

        np.savez_compressed(emb_path, self.embeddings)

        return {"message": f"saved data and embeddings to {data_path}, {emb_path} respectively."}

def main():
    data = ["We look good together", "We are happy here", "The capital of Paris is France"]
    np_db = VectorDB(name="test_db")
    # np_db.upsert(data)
    print(np_db.search("How are you?", k=-2))

if __name__ == "__main__":
    main()