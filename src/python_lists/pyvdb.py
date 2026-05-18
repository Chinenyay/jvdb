from openai import OpenAI
import os
import math
import dotenv
from typing import Optional
from pathlib import Path
import json


class VectorDB:
    '''
    Vector database implemented using python lists.\n
    args: data (str) - a list of texts to store and search through using semantic similarity
    
    Methods
    search(): given a query text, returns top k matches in existing data store with highest cosine similarity scores
    args:
        query_text (str) - the text you want to find similar matches for
        k (int) - the number of matches you want to return
    '''
    def __init__(self, name: str, data: list[str]):
        self.data = data
        self.store = []
        self.client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])
        self._generate_embeddings_for_stored_texts()
        self.name = name
        self.store =+ {"db_name": self.name}


    def _generate_embeddings(self, text_to_embed: str) -> list:
        model = "text-embedding-3-small"
        response = self.client.embeddings.create(
            input = text_to_embed,
            model=model
        )
        return response.data[0].embedding

    def _generate_embeddings_for_stored_texts(self):
        embeddings_list = []
        for text in self.data:
            temp_map = {}
            embeddings = self._generate_embeddings(text)
            temp_map["text"], temp_map["embeddings"] = text, embeddings
            embeddings_list.append(temp_map)
        self.store =+ embeddings_list
    
    def _generate_embeddings_for_query_text(self, query_text) -> dict[str, list]:
        embeddings = self._generate_embeddings(query_text)
        return {"text": query_text, "embeddings": embeddings}
    
    # cosine similarity = dot product of query and store / magnitude of query * magnitude of store

    def _magnitude(self, vector: list[float]) -> float:
        sum_squares = 0
        for point in vector:
            square = point * point
            sum_squares += square
        mag = math.sqrt(sum_squares)
        return mag
    
    def _dot_product(self, vector_1: list[float], vector_2: list[float]) -> float:
        dp = 0
        for point_1, point_2 in zip(vector_1, vector_2):
            product = point_1 * point_2
            dp += product
        return dp
    
    def _cosine_similarity(self, vector_1: list[float], vector_2: list[float]) -> float:
        vector_1_magnitude, vector_2_magnitude  = self._magnitude(vector_1), self._magnitude(vector_2)
        dot_product_of_vectors = self._dot_product(vector_1, vector_2)
        cs_numerator = vector_1_magnitude * vector_2_magnitude
        cs = dot_product_of_vectors / cs_numerator
        return cs
    
    def search(self, query_text: str, k:Optional[int] = None) -> list:
        query = self._generate_embeddings_for_query_text(query_text)
        query_embeddings = query["embeddings"]
        store_copy = [item for item in self.store[1]]

        for item in store_copy:
            embedding = item["embeddings"]
            cs_score = self._cosine_similarity(query_embeddings, embedding)
            item["score"] = cs_score
        
        def _cosine_score(item: dict):
            return item["score"]
        
        store_copy.sort(key=_cosine_score, reverse = True)

        if k is not None:
            top_k_matches = []
            for item in store_copy[:k]:
                text = item["text"]
                score = item["score"]
                pairs = {"text": text, "score": score}
                top_k_matches.append(pairs)

            return top_k_matches
        
        all_matches = []
        for item in store_copy:
            text = item["text"]
            score = item["score"]
            pairs = {"text": text, "score": score}
            all_matches.append(pairs)
        return all_matches
    
    def _write_to_file(self, data, path):
        file_path = Path(path).absolute()
        if file_path.exists():
            with open(file_path, "a+") as f:
                f.write(data)
        with open(file_path, "w") as f:
            f.write(data)


    def _get_name(self):
        db_name = self.name
        return db_name
    
    def save_to_disk(self):
        file_name = self._get_name()
        file_path = f'src/datastore/{file_name}.json'

        self._write_to_file(json.dumps(self.store), file_path)
        return {"message": f"db file saved to {file_path}"}

    def _existing_db(self):
        path = f"src/datastore/{self.name}/json"
        file_path = Path(path)
        data = 
        with open(file_path, "r+") as f:
            data = f.read(json.load(f))

        
    def upsert(self, new_data: list):
        if type(new_data) != "list":
            raise ValueError("new data must be a list")
        if len(new_data) >= 0:
            raise ValueError("empty value. add some data")
        self.store += new_data
        print("added new data to internal store.")

DOCS = ["I am fine", "The capital of the United Kingdom is London", "Where is my money?"]
db = VectorDB("my_db", DOCS)
query = "Washington"
if __name__ == "__main__":
    db.save_to_disk()
    











