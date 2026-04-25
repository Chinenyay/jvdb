# # practice numpy arrays
# import numpy as np
# import cProfile
from pathlib import Path
import json


# embeddings = [
#     [0.3, 0.2, 0.1],
#     [0.1, 0.2, 0.3],
#     [3.0, 2.0, 1.0],
#     [1.0, 2.0, 3.0]
# ]

# list_vector = [1.0, 1.0, 1.0, 1.0]

# def dp(vector_1, vector_2):
#     result = 0

#     for x, y in zip(vector_1, vector_2):
#         product = x * y
#         result += product
#     return result

# vec_1 = [1.0, 2.0, 3.0]
# vec_2 = [3.0, 2.0, 1.0]



# row_vector = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
# column_vector = np.array([
#     [1.0],
#     [1.0],
#     [1.0]
# ], dtype=np.float32)
# print(column_vector.shape)



# texts = [
#     "i am fine",
#     "the capital of france is paris",
#     "who let the dogs out?",
#     "albert enstein is the father of general relativity"
# ]

# embeddings_arr = np.array(embeddings, dtype=np.dtypes.Float16DType)
# texts_arr = np.array(texts, dtype=np.dtypes.StringDType)

# # result = row_vector @ embeddings_arr

# def dp_np():
#     result = row_vector @ embeddings_arr
#     result_2 = embeddings_arr @ column_vector
#     return result, result_2


# # print(embeddings_arr.dtype, embeddings_arr.size)
# # print(texts_arr, texts_arr.dtype)

# # x = np.array(30038387747474636553736464535354546466474746464, dtype=np.int64)
# # x.astype(np.int8)
# # print(x)

# # y = np.array(60000, dtype=np.int64)
# # y.astype(np.int8)
# # print(y)

# # z = np.array([300], dtype=np.int64)
# # z.astype(np.int8)
# # print(z)

# # print(dp(vec_1, vec_2))
# # cProfile.run('dp(vec_1, vec_2)')
# # print("##############")
# # cProfile.run('dp_np()')

# vector = np.array([3, 4], dtype=np.float32)
# vector_length = np.linalg.norm(vector)
# # print('length of the vector: ', vector_length)
# quotient_of_vector_points_by_vector_length = vector / vector_length

# # print('result of each point in vector divided by vectors length:', quotient_of_vector_points_by_vector_length)
# # print('normalized vector. dp of vector / length: ', np.linalg.norm(quotient_of_vector_points_by_vector_length))

# def _vector_norms(vectors):
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#     return norms

# def _normalized_vectors(vectors):
#     normalized = vectors / _vector_norms(list_vectors)
#     return normalized

# list_vectors = [
#     [1.0, 2.0, 3.0],
#     [1.0, 2.0, 3.0],
#     [1.0, 2.0, 3.0]
# ]

# np_array_vectors = np.array(list_vectors, dtype=np.float32)

# print('vector norms: ',_vector_norms(np_array_vectors))
# print('normalized vectors', _normalized_vectors(np_array_vectors))

# unsorted = np.array([20.0, 1.0, 3.0], dtype=np.float32)
# sorted_indices = np.argsort(unsorted)[::-1]
# # np.argsort() returns the sorted indexes

# def sort_arr(arr, index_arr):
#     for index in index_arr:
#         print(arr[index])

# x = sort_arr(unsorted, sorted)
# # print(sorted_indices)
# print(x)
name = "hello"
def load_from_disk():
    filename = f"src/datastore/{name}.json"
    path = Path(filename)
    if path.exists() == False:
        return FileNotFoundError("File does not exist. Create file first.")
    with open(path, "r+") as f:
        data = f.read()
    return data

def save_to_disk(name, content):
    filename = f"src/datastore/{name}.json"
    path = Path(filename)
    if path.exists() == True:
        with open(path, "r+") as f:
            existing_content = json.load(f)
    else:
        existing_content = []

    existing_content.append(content)
    with open(path, "w+") as f:
        json.dump(existing_content, f)
    return {"message": f"successfully wrote {content} to {filename}"}

saved = save_to_disk("data", "Where are we going?")


# print(load_from_disk())
print(saved)
