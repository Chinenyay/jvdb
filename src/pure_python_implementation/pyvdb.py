from openai import OpenAI
import os
import math
import dotenv

class VectorDB:
    '''
    Vector database implemented using python lists.\n
    args: data (str) - a list of texts to store and search through using semantic similarity 
    '''
    def __init__(self, data: list[str]):
        self.data = data
        self.store = []

    _client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])

    def _generate_embeddings(self, text_to_embed: str) -> list:
        model = "text-embedding-3-small"
        response = self._client.embeddings.create(
            input = text_to_embed,
            model=model
        )
        return response.data[0].embedding

    def _generate_embeddings_for_stored_texts(self):
        for text in self.data:
            temp_map = {}
            embeddings = self._generate_embeddings(text)
            temp_map["text"], temp_map["embeddings"] = text, embeddings
            self.store.append(temp_map)
    
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
    
    def search(self, query_text: str, k:int):
        matches = []
        query = self._generate_embeddings_for_query_text(query_text)
        query_embeddings = query["embeddings"]
        self._generate_embeddings_for_stored_texts()
        store_copy = [item for item in self.store]

        for item in store_copy:
            embedding = item["embeddings"]
            cs_score = self._cosine_similarity(query_embeddings, embedding)
            item["score"] = cs_score
        
        def _cosine_score(item: dict):
            return item["score"]
        
        store_copy.sort(key=_cosine_score, reverse = True)

        for item in store_copy[:k]:
            text = item["text"]
            score = item["score"]
            pairs = {"text": text, "score": score}
            matches.append(pairs)

        return matches

if __name__ == "__main__":
    DOCS = ["I am fine", "The capital of the United Kingdom is London", "Where is my money?"]
    db = VectorDB(DOCS)
    query = "Washington"
    result = db.search(query, 2)
    print(result)











