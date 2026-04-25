# store = [
#     {
#         "text": "How are you?",
#         "embeddings": [1.0, 2.0, 3.0]
#     },
#     {
#         "text": "Where are you?",
#         "embeddings": [1.0, 2.0, 3.0]
#     }
# ]

# store_embeddings = [item["embeddings"] for item in store]
# print(store_embeddings)

from pathlib import Path
def write_to_file(data, path):
    file_path = Path(path).absolute()
    with open(file_path, "a+") as f:
        f.write(data)

# write_to_file("i am fine?", "src/datastore/hello.py")
# import json
# with open("src/datastore/my_db.json") as f:
#     data = json.load(f)
# print(len(data))

inner_list = [1, 2, 3]
inner_list_2 = [4, 5, 6]
inner_dict = {"text": "hello"}
outer_list = []
outer_list += inner_dict, inner_list_2
print(outer_list[1])