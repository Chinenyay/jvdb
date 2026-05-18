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

# inner_list = [1, 2, 3]
# inner_list_2 = [4, 5, 6]
# inner_dict = {"text": "hello"}
# outer_list = []
# outer_list += inner_dict, inner_list_2
# print(outer_list[1])

# a_list = []
# print(len(a_list))

# def is_dup(a, b) -> bool:
#     if a == b:
#         return True
#     else:
#         return False


# if not is_dup(2, 3):
#     print("No duplicate")
# if is_dup(2, 2):
#     print("These are duplicates")

# data = [
#     {1: "one"},
#     {2: "two"},
#     {3: "three"}
# ]

# for d in data:
#     for key, value in d.items():
#         print(value)