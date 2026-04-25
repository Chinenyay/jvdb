# NumPy Tutorial for Building `pyvdb.py` From Scratch

This tutorial teaches the NumPy concepts that matter for your current vector
database implementation in `src/pure_python_implementation/pyvdb.py`.

Your current version uses normal Python lists to:

1. Store texts.
2. Generate one embedding per text.
3. Compute cosine similarity between a query embedding and every stored
   embedding.
4. Sort the results by similarity score.
5. Return the top `k` matches.

That is the right first implementation. NumPy helps with the next version
because embeddings are numeric vectors, and NumPy is designed for fast numeric
arrays.

The key shift is:

```python
# Pure Python mental model
vector = [0.2, -0.1, 0.9]

# NumPy mental model
vector = np.array([0.2, -0.1, 0.9])
```

For a vector database, this gets even more useful when you store many vectors
together:

```python
# 3 documents, each with 4 embedding values
embeddings = np.array([
    [0.10, 0.20, 0.30, 0.40],
    [0.90, 0.10, 0.20, 0.30],
    [0.05, 0.80, 0.10, 0.25],
])
```

That 2D array is the core data structure for a NumPy-backed vector database.

---

## 1. What NumPy Gives You

NumPy gives you an array type called `ndarray`.

An `ndarray` is like a Python list, but optimized for numbers.

Python lists can hold mixed values:

```python
items = ["hello", 42, True, [1, 2, 3]]
```

NumPy arrays usually hold one kind of value:

```python
import numpy as np

vector = np.array([0.1, 0.2, 0.3])
```

That restriction is useful. Because NumPy knows every item is numeric, it can
perform math across the whole array quickly.

In your current `pyvdb.py`, these functions are pure Python loops:

```python
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
```

With NumPy, those become:

```python
import numpy as np

magnitude = np.linalg.norm(vector)
dot_product = np.dot(vector_1, vector_2)
```

The NumPy version is shorter, clearer, and much faster when there are many
vectors.

---

## 2. Installing NumPy in This Project

Your `pyproject.toml` currently lists `openai` and `dotenv`, but not NumPy.

Since this project uses `uv`, you can add NumPy with:

```bash
uv add numpy
```

Then in Python:

```python
import numpy as np
```

The alias `np` is the normal convention. Almost every NumPy project uses it.

---

## 3. From Python Lists to NumPy Arrays

Your OpenAI embedding call returns a normal Python list:

```python
embedding = response.data[0].embedding
```

That value is shaped like this:

```python
[0.0123, -0.044, 0.108, ...]
```

To use it with NumPy:

```python
embedding_array = np.array(embedding)
```

For vector databases, you usually want floating point numbers:

```python
embedding_array = np.array(embedding, dtype=np.float32)
```

`dtype` means "data type".

Common dtypes:

```python
np.float64  # default Python-like floating point precision
np.float32  # smaller, often enough for embeddings
np.int64    # integers
np.bool_    # booleans
```

For embeddings, `np.float32` is a practical default because it uses less memory
than `np.float64`.

Example:

```python
import numpy as np

embedding = [0.1, 0.2, 0.3]
arr = np.array(embedding, dtype=np.float32)

print(arr)
print(arr.dtype)
```

Output:

```text
[0.1 0.2 0.3]
float32
```

---

## 4. Shape: The Most Important NumPy Concept

Every NumPy array has a `shape`.

```python
vector = np.array([0.1, 0.2, 0.3])
print(vector.shape)
```

Output:

```text
(3,)
```

That means:

- This is a 1D array.
- It has 3 values.

A matrix is a 2D array:

```python
embeddings = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
])

print(embeddings.shape)
```

Output:

```text
(3, 3)
```

That means:

- 3 rows.
- 3 columns.

For your vector database:

```text
embeddings.shape == (number_of_documents, embedding_dimension)
```

If you have 100 documents and each embedding has 1536 numbers:

```text
embeddings.shape == (100, 1536)
```

You do not need to hard-code the embedding dimension. You can let NumPy infer it
from the embeddings returned by the model.

Why shape matters:

```python
query_embedding.shape      # (embedding_dimension,)
stored_embeddings.shape    # (number_of_documents, embedding_dimension)
```

Those shapes tell you whether the math can work.

---

## 5. Vectors, Matrices, and Your Vector Database

In your current implementation, each stored item looks like this:

```python
{
    "text": text,
    "embeddings": embeddings,
}
```

That is beginner-friendly, but it mixes text and numeric data in one list of
dictionaries.

For a NumPy version, use two separate structures:

```python
self.texts = [
    "I am fine",
    "The capital of the United Kingdom is London",
    "Where is my money?",
]

self.embeddings = np.array([
    [...],
    [...],
    [...],
], dtype=np.float32)
```

This separation is important:

- Text stays in a Python list.
- Embeddings go into a NumPy array.
- The row index connects them.

Example:

```python
self.texts[0]          # first text
self.embeddings[0]     # first text's embedding

self.texts[1]          # second text
self.embeddings[1]     # second text's embedding
```

This is a core vector database idea.

Instead of storing records like this:

```text
[
  {"text": "A", "embedding": [0.1, 0.2]},
  {"text": "B", "embedding": [0.3, 0.4]},
]
```

You store columns:

```text
texts      = ["A", "B"]
embeddings = [[0.1, 0.2],
              [0.3, 0.4]]
```

This makes similarity search much easier.

---

## 6. Creating a 2D Embedding Matrix

Imagine your current method:

```python
def _generate_embeddings_for_stored_texts(self):
    for text in self.data:
        temp_map = {}
        embeddings = self._generate_embeddings(text)
        temp_map["text"], temp_map["embeddings"] = text, embeddings
        self.store.append(temp_map)
```

In a NumPy version, you can collect the embeddings first:

```python
all_embeddings = []

for text in self.data:
    embedding = self._generate_embeddings(text)
    all_embeddings.append(embedding)
```

Then convert the list of lists into a 2D NumPy array:

```python
self.embeddings = np.array(all_embeddings, dtype=np.float32)
```

If each embedding has the same length, the result is a clean matrix:

```python
print(self.embeddings.shape)
```

Example output:

```text
(3, 1536)
```

This means:

- 3 stored texts.
- Each text has a 1536-number embedding.

Again, you do not need to hard-code `1536`. The important thing is that every
stored embedding has the same length.

---

## 7. Elementwise Math

NumPy can apply math to every value in an array.

With Python lists:

```python
vector = [1, 2, 3]

squares = []
for value in vector:
    squares.append(value * value)
```

With NumPy:

```python
import numpy as np

vector = np.array([1, 2, 3])
squares = vector * vector
```

Output:

```text
[1 4 9]
```

This matters because your `_magnitude` method does:

```python
point * point
```

for every point in the vector.

In NumPy:

```python
squares = vector * vector
sum_squares = np.sum(squares)
magnitude = np.sqrt(sum_squares)
```

That is the manual NumPy version.

The shorter version:

```python
magnitude = np.linalg.norm(vector)
```

---

## 8. Dot Product

Your current dot product:

```python
def _dot_product(self, vector_1: list[float], vector_2: list[float]) -> float:
    dp = 0
    for point_1, point_2 in zip(vector_1, vector_2):
        product = point_1 * point_2
        dp += product
    return dp
```

The dot product multiplies matching positions and adds the results.

Example:

```text
vector_1 = [1, 2, 3]
vector_2 = [4, 5, 6]

dot = (1 * 4) + (2 * 5) + (3 * 6)
dot = 4 + 10 + 18
dot = 32
```

In NumPy:

```python
import numpy as np

vector_1 = np.array([1, 2, 3])
vector_2 = np.array([4, 5, 6])

dot = np.dot(vector_1, vector_2)
print(dot)
```

Output:

```text
32
```

You can also use the `@` operator:

```python
dot = vector_1 @ vector_2
```

For your project, `@` becomes very useful when comparing one query vector
against every stored embedding.

---

## 9. Magnitude and Norm

Your current magnitude:

```python
def _magnitude(self, vector: list[float]) -> float:
    sum_squares = 0
    for point in vector:
        square = point * point
        sum_squares += square
    mag = math.sqrt(sum_squares)
    return mag
```

This is also called the vector's norm.

For this vector:

```text
[3, 4]
```

The norm is:

```text
sqrt((3 * 3) + (4 * 4))
sqrt(9 + 16)
sqrt(25)
5
```

In NumPy:

```python
vector = np.array([3, 4])
norm = np.linalg.norm(vector)
print(norm)
```

Output:

```text
5.0
```

For one vector:

```python
np.linalg.norm(vector)
```

For many vectors stored as rows in a matrix:

```python
np.linalg.norm(embeddings, axis=1)
```

`axis=1` means "do this across each row".

Example:

```python
embeddings = np.array([
    [3, 4],
    [5, 12],
])

norms = np.linalg.norm(embeddings, axis=1)
print(norms)
```

Output:

```text
[ 5. 13.]
```

This is one of the most important tricks for vector search.

---

## 10. Understanding `axis`

`axis` is one of the most confusing beginner NumPy concepts.

Start with this matrix:

```python
arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

Its shape is:

```text
(2, 3)
```

That means:

- Axis 0 is the row direction.
- Axis 1 is the column direction.

When you sum with `axis=0`:

```python
np.sum(arr, axis=0)
```

Output:

```text
[5 7 9]
```

That means:

```text
[1 + 4, 2 + 5, 3 + 6]
```

It collapsed the rows.

When you sum with `axis=1`:

```python
np.sum(arr, axis=1)
```

Output:

```text
[ 6 15]
```

That means:

```text
[1 + 2 + 3, 4 + 5 + 6]
```

It collapsed the columns and gave one result per row.

For vector databases:

```python
np.linalg.norm(embeddings, axis=1)
```

means:

```text
For each row embedding, compute one norm.
```

If `embeddings.shape == (100, 1536)`, then:

```python
np.linalg.norm(embeddings, axis=1).shape
```

is:

```text
(100,)
```

One norm per stored document.

---

## 11. Cosine Similarity

Your current comment says:

```python
# cosine similarity = dot product of query and store / magnitude of query * magnitude of store
```

The formula is:

```text
cosine_similarity = dot_product(vector_1, vector_2) / (norm(vector_1) * norm(vector_2))
```

In pure Python, your code computes:

```python
vector_1_magnitude = self._magnitude(vector_1)
vector_2_magnitude = self._magnitude(vector_2)
dot_product_of_vectors = self._dot_product(vector_1, vector_2)
cs_denominator = vector_1_magnitude * vector_2_magnitude
cs = dot_product_of_vectors / cs_denominator
```

In NumPy for two vectors:

```python
def cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    dot = np.dot(vector_1, vector_2)
    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)
    return dot / (norm_1 * norm_2)
```

That is a direct translation of your current implementation.

But a vector database needs one query against many stored embeddings.

---

## 12. Comparing One Query Against All Stored Embeddings

Suppose:

```python
query.shape       # (1536,)
embeddings.shape  # (100, 1536)
```

You want 100 cosine similarity scores.

One score for each stored document.

### Step 1: Dot product with every row

Use:

```python
dots = embeddings @ query
```

If:

```text
embeddings.shape == (100, 1536)
query.shape == (1536,)
```

then:

```text
dots.shape == (100,)
```

That gives one dot product per stored embedding.

This replaces this loop:

```python
for item in store_copy:
    embedding = item["embeddings"]
    cs_score = self._cosine_similarity(query_embeddings, embedding)
    item["score"] = cs_score
```

### Step 2: Norm of the query

```python
query_norm = np.linalg.norm(query)
```

This is one number.

### Step 3: Norm of every stored embedding

```python
embedding_norms = np.linalg.norm(embeddings, axis=1)
```

This returns one number per stored row.

### Step 4: Compute all cosine scores

```python
scores = dots / (embedding_norms * query_norm)
```

Now:

```text
scores.shape == (100,)
```

Each score lines up with `self.texts`:

```python
self.texts[0]  # score is scores[0]
self.texts[1]  # score is scores[1]
self.texts[2]  # score is scores[2]
```

Complete function:

```python
def cosine_similarity_many(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    dots = embeddings @ query
    query_norm = np.linalg.norm(query)
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    return dots / (embedding_norms * query_norm)
```

This is the central NumPy operation for your vector database.

---

## 13. Broadcasting

Broadcasting is how NumPy combines arrays with different shapes.

In this line:

```python
scores = dots / (embedding_norms * query_norm)
```

The shapes are:

```text
dots.shape            == (100,)
embedding_norms.shape == (100,)
query_norm            == scalar
```

`query_norm` is one number, but NumPy automatically applies it to all 100
items.

That is broadcasting.

Simple example:

```python
arr = np.array([10, 20, 30])
result = arr / 10
print(result)
```

Output:

```text
[1. 2. 3.]
```

You did not write a loop. NumPy broadcast `10` across the whole array.

Another example:

```python
scores = np.array([0.2, 0.5, 0.9])
boosted = scores + 0.1
```

Output:

```text
[0.3 0.6 1. ]
```

For your vector database, broadcasting lets you divide every dot product by the
query norm and the matching stored vector norm.

---

## 14. Sorting and Top `k`

Your current search sorts dictionaries:

```python
store_copy.sort(key=_cosine_score, reverse=True)
```

With NumPy, you sort indexes instead of dictionaries.

Imagine:

```python
texts = ["A", "B", "C"]
scores = np.array([0.20, 0.95, 0.50])
```

You want the indexes of the highest scores.

Use `np.argsort`:

```python
sorted_indexes = np.argsort(scores)
print(sorted_indexes)
```

Output:

```text
[0 2 1]
```

Why?

```text
scores[0] == 0.20
scores[2] == 0.50
scores[1] == 0.95
```

That is ascending order.

For descending order:

```python
sorted_indexes = np.argsort(scores)[::-1]
print(sorted_indexes)
```

Output:

```text
[1 2 0]
```

Top `k`:

```python
k = 2
top_indexes = np.argsort(scores)[::-1][:k]
```

Then build results:

```python
matches = []

for index in top_indexes:
    matches.append({
        "text": texts[index],
        "score": float(scores[index]),
    })
```

`float(scores[index])` converts NumPy's float type into a normal Python float,
which is nicer for printing and JSON.

---

## 15. Faster Top `k` With `argpartition`

For a small learning project, `np.argsort(scores)[::-1][:k]` is perfect.

For larger databases, sorting every score can be wasteful. If you only need the
top 5 results out of 1 million scores, you do not need a full sort.

NumPy has `np.argpartition`:

```python
top_indexes_unsorted = np.argpartition(scores, -k)[-k:]
```

This finds the top `k` indexes, but does not fully sort them.

To sort just those top results:

```python
top_indexes = top_indexes_unsorted[np.argsort(scores[top_indexes_unsorted])[::-1]]
```

Beginner version:

```python
top_indexes = np.argsort(scores)[::-1][:k]
```

More scalable version:

```python
top_indexes_unsorted = np.argpartition(scores, -k)[-k:]
top_indexes = top_indexes_unsorted[np.argsort(scores[top_indexes_unsorted])[::-1]]
```

Use `argsort` first while learning. Switch to `argpartition` when you are
working with enough data for sorting to become a bottleneck.

---

## 16. Boolean Masks for Filtering

Your README says you want to add a filter step in search.

NumPy is useful for filtering because it supports boolean masks.

Example:

```python
scores = np.array([0.20, 0.95, 0.50])
mask = scores >= 0.5

print(mask)
```

Output:

```text
[False  True  True]
```

Use the mask:

```python
filtered_scores = scores[mask]
print(filtered_scores)
```

Output:

```text
[0.95 0.5 ]
```

For a vector database, you may not filter only by score. You may filter by
metadata.

Example:

```python
texts = ["A", "B", "C"]
categories = np.array(["note", "city", "note"])
scores = np.array([0.20, 0.95, 0.50])

mask = categories == "note"
candidate_indexes = np.where(mask)[0]
```

Output:

```text
[0 2]
```

Then search only those rows:

```python
candidate_embeddings = embeddings[candidate_indexes]
candidate_scores = cosine_similarity_many(query, candidate_embeddings)
```

The scores now line up with `candidate_indexes`, not the original full list.

To return original texts:

```python
top_local_indexes = np.argsort(candidate_scores)[::-1][:k]

matches = []
for local_index in top_local_indexes:
    original_index = candidate_indexes[local_index]
    matches.append({
        "text": texts[original_index],
        "score": float(candidate_scores[local_index]),
    })
```

Filtering is more advanced, but the key concept is simple:

```python
mask = condition_that_returns_true_or_false_per_row
filtered_array = array[mask]
```

---

## 17. Avoiding Repeated Work

Your current `search` method calls:

```python
self._generate_embeddings_for_stored_texts()
```

inside every search.

That means stored document embeddings are regenerated every time you search.

For a vector database, you usually want:

1. Generate stored embeddings once.
2. Keep them in memory.
3. Generate only the query embedding during search.
4. Compare the query against the stored matrix.

A NumPy version could have a `build` method:

```python
def build(self) -> None:
    all_embeddings = []

    for text in self.texts:
        embedding = self._generate_embeddings(text)
        all_embeddings.append(embedding)

    self.embeddings = np.array(all_embeddings, dtype=np.float32)
```

Then:

```python
db = VectorDB(DOCS)
db.build()
results = db.search("Washington", k=2)
```

This separates setup from search.

That is a major design improvement, not just a NumPy improvement.

---

## 18. Pre-normalizing Embeddings

This section is easier if you separate the idea into smaller pieces:

1. What is a vector?
2. What is a norm?
3. What does normalizing do?
4. Why does that make search faster?
5. Why does `keepdims=True` matter?

### A vector is just a list of numbers

For this tutorial, a vector means one row of numbers:

```python
vector = np.array([3, 4], dtype=np.float32)
```

An embedding is also a vector. It usually has many more numbers, but the idea is
the same:

```python
embedding = np.array([0.12, -0.45, 0.88, 0.03], dtype=np.float32)
```

You can think of each number as one coordinate. The vector `[3, 4]` means:

```text
3 steps in the first direction
4 steps in the second direction
```

Real text embeddings may have hundreds or thousands of directions, but NumPy
treats them the same way.

### The norm is the vector's length

The norm tells you how long the vector is.

For this vector:

```python
vector = np.array([3, 4], dtype=np.float32)
```

The norm is:

```text
sqrt(3 * 3 + 4 * 4)
sqrt(9 + 16)
sqrt(25)
5
```

In NumPy:

```python
np.linalg.norm(vector)
```

Output:

```text
5.0
```

So `[3, 4]` has length `5`.

### Normalizing means making the length equal to 1

To normalize a vector, divide every number in the vector by the vector's norm.

For `[3, 4]`, the norm is `5`, so:

```text
[3, 4] / 5
```

becomes:

```text
[3 / 5, 4 / 5]
```

which is:

```text
[0.6, 0.8]
```

In NumPy:

```python
vector = np.array([3, 4], dtype=np.float32)
norm = np.linalg.norm(vector)
normalized = vector / norm

print(normalized)
print(np.linalg.norm(normalized))
```

Output:

```text
[0.6 0.8]
1.0
```

The normalized vector still points in the same direction as `[3, 4]`. It has
just been scaled down so its length is `1`.

That is why normalized vectors are called unit vectors.

A unit vector means:

```text
a vector whose norm is 1
```

### Cosine similarity compares direction

Cosine similarity measures whether two vectors point in similar directions.

The formula is:

```text
dot_product(a, b) / (norm(a) * norm(b))
```

The dot product means:

```text
multiply matching numbers, then add the results
```

Example:

```python
a = np.array([1, 2], dtype=np.float32)
b = np.array([3, 4], dtype=np.float32)

dot = np.dot(a, b)
print(dot)
```

This computes:

```text
(1 * 3) + (2 * 4)
3 + 8
11
```

Output:

```text
11.0
```

Cosine similarity uses the dot product, but it also divides by the lengths of
the vectors. That division removes the effect of vector size, so the score is
about direction instead of raw length.

### Why normalizing first simplifies cosine similarity

Cosine similarity normally does this:

```text
dot_product(a, b) / (norm(a) * norm(b))
```

But if `a` and `b` are already normalized, both norms are `1`.

So the formula becomes:

```text
dot_product(a, b) / (1 * 1)
```

That is the same as:

```text
dot_product(a, b)
```

So after normalization:

```text
cosine similarity == dot product
```

That is the key idea.

### Why this helps vector search

Imagine you have 10,000 stored document embeddings.

Without pre-normalizing, every search has to repeatedly do work like this:

```text
for every stored document:
    calculate dot product
    calculate query norm
    calculate stored document norm
    divide by both norms
```

But the stored document embeddings do not change between searches. Their norms
also do not change.

So you can do this once during setup:

```python
normalized_stored_embeddings = normalize_rows(stored_embeddings)
```

Then for each search, you only normalize the query once:

```python
normalized_query = query / np.linalg.norm(query)
```

Now every stored vector has norm `1`, and the query vector has norm `1`, so
search can use a dot product directly:

```python
scores = normalized_stored_embeddings @ normalized_query
```

### A matrix is a table of vectors

When you store many embeddings together, you have a 2D NumPy array.

A 2D array is often called a matrix.

For vector search, the matrix usually means:

```text
one row per document
one column per embedding number
```

Example:

```python
embeddings = np.array([
    [3, 4],
    [10, 0],
    [0, 2],
], dtype=np.float32)
```

This matrix has 3 rows and 2 columns:

```text
row 0: [3, 4]   document 0
row 1: [10, 0]  document 1
row 2: [0, 2]   document 2
```

The shape is:

```python
print(embeddings.shape)
```

Output:

```text
(3, 2)
```

That means:

```text
3 documents
2 numbers per embedding
```

In a real embedding system, the shape might be:

```text
(10000, 1536)
```

That would mean:

```text
10000 documents
1536 numbers per embedding
```

### Normalizing every row

For one vector, you do this:

```python
norm = np.linalg.norm(vector)
normalized = vector / norm
```

For a whole matrix of embeddings, you want one norm per row:

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized_embeddings = embeddings / norms
```

The `axis=1` part means:

```text
calculate across each row
```

For this matrix:

```python
embeddings = np.array([
    [3, 4],
    [10, 0],
    [0, 2],
], dtype=np.float32)
```

the row norms are:

```text
norm of [3, 4]   = 5
norm of [10, 0]  = 10
norm of [0, 2]   = 2
```

So:

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
print(norms)
```

Output:

```text
[[ 5.]
 [10.]
 [ 2.]]
```

Then:

```python
normalized_embeddings = embeddings / norms
print(normalized_embeddings)
```

Output:

```text
[[0.6 0.8]
 [1.  0. ]
 [0.  1. ]]
```

Each row was divided by its own norm:

```text
[3, 4]   / 5  = [0.6, 0.8]
[10, 0]  / 10 = [1.0, 0.0]
[0, 2]   / 2  = [0.0, 1.0]
```

### Why `keepdims=True` matters

When NumPy calculates one norm per row, the result can have two different
shapes.

Without `keepdims=True`:

```python
norms.shape == (number_of_documents,)
```

With it:

```python
norms.shape == (number_of_documents, 1)
```

Using the small example:

```python
embeddings.shape == (3, 2)
```

Without `keepdims=True`, the norms shape is:

```python
norms.shape == (3,)
```

That means NumPy sees it like this:

```text
[5, 10, 2]
```

With `keepdims=True`, the norms shape is:

```python
norms.shape == (3, 1)
```

That means NumPy sees it like this:

```text
[
    [5],
    [10],
    [2],
]
```

That second shape lines up with the embedding rows:

```text
embeddings.shape == (number_of_documents, embedding_dimension)
norms.shape      == (number_of_documents, 1)
```

NumPy divides each row by that row's norm.

This is called broadcasting.

Broadcasting means NumPy stretches a smaller array across a bigger array when
their shapes are compatible.

In this case, NumPy treats the norms like this:

```text
[
    [5, 5],
    [10, 10],
    [2, 2],
]
```

It does not actually create that bigger array in memory. It just behaves as if
it did.

That lets this line work:

```python
normalized_embeddings = embeddings / norms
```

### What `@` means

This line:

```python
scores = normalized_stored_embeddings @ normalized_query_embedding
```

means:

```text
dot every stored embedding row with the query embedding
```

Example:

```python
normalized_stored_embeddings = np.array([
    [0.6, 0.8],
    [1.0, 0.0],
    [0.0, 1.0],
], dtype=np.float32)

normalized_query_embedding = np.array([0.6, 0.8], dtype=np.float32)

scores = normalized_stored_embeddings @ normalized_query_embedding
print(scores)
```

Output:

```text
[1.  0.6 0.8]
```

That means:

```text
document 0 score: 1.0
document 1 score: 0.6
document 2 score: 0.8
```

Document 0 scores highest because its normalized vector is exactly the same as
the query vector.

### The final search pattern

During setup, normalize the stored embeddings once:

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized_embeddings = embeddings / norms
```

During search, normalize the query once:

```python
query = np.array(query_embedding, dtype=np.float32)
query = query / np.linalg.norm(query)
```

Then calculate all similarity scores at once:

```python
scores = self.embeddings @ query
```

This works if `self.embeddings` already contains normalized stored embeddings.

So the simple high-performance brute force search pattern is:

```python
scores = normalized_stored_embeddings @ normalized_query_embedding
```

Read that as:

```text
Compare the normalized query against every normalized stored document.
Return one score per document.
```

---

## 19. Handling Zero Vectors

Cosine similarity divides by vector norms.

If a vector has norm `0`, division fails:

```python
zero = np.array([0, 0, 0], dtype=np.float32)
np.linalg.norm(zero)
```

Output:

```text
0.0
```

Then:

```python
zero / np.linalg.norm(zero)
```

would divide by zero.

Embeddings from a real embedding model should usually not be zero vectors, but
it is still good practice to guard against this.

```python
def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms
```

For a single query:

```python
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
```

`np.where` means:

```python
np.where(condition, value_if_true, value_if_false)
```

So:

```python
norms = np.where(norms == 0, 1, norms)
```

means:

```text
If a norm is 0, replace it with 1. Otherwise keep the norm.
```

---

## 20. Indexing NumPy Arrays

You will use indexing constantly.

One row:

```python
first_embedding = embeddings[0]
```

Multiple rows:

```python
selected = embeddings[[0, 2, 5]]
```

A slice:

```python
first_ten = embeddings[:10]
```

One column:

```python
first_dimension = embeddings[:, 0]
```

Meaning:

```text
embeddings[:, 0]
           |  |
           |  first column
           all rows
```

For vector search, the most important indexing pattern is:

```python
top_indexes = np.argsort(scores)[::-1][:k]
top_scores = scores[top_indexes]
top_embeddings = embeddings[top_indexes]
```

Then map indexes back to text:

```python
for index in top_indexes:
    print(texts[index], scores[index])
```

---

## 21. Copies vs Views

NumPy sometimes returns a copy and sometimes returns a view.

A view points at the same underlying data.

Example:

```python
arr = np.array([1, 2, 3, 4])
view = arr[:2]
view[0] = 99

print(arr)
```

Output:

```text
[99  2  3  4]
```

Changing `view` changed `arr`.

This matters when you slice arrays and mutate them.

If you want an independent copy:

```python
copy = arr[:2].copy()
```

For your vector database search, you usually do not need to mutate stored
embeddings. That makes this simpler:

- Store embeddings once.
- Treat them as read-only during search.
- Create new arrays for query embeddings and scores.

---

## 22. Saving and Loading Embeddings

A vector database should not need to call the embedding API every time it
starts.

NumPy can save arrays to disk.

Save only embeddings:

```python
np.save("embeddings.npy", self.embeddings)
```

Load them:

```python
self.embeddings = np.load("embeddings.npy")
```

To save texts and embeddings together, use `np.savez`:

```python
np.savez(
    "vector_store.npz",
    texts=np.array(self.texts),
    embeddings=self.embeddings,
)
```

Load:

```python
data = np.load("vector_store.npz")
self.texts = data["texts"].tolist()
self.embeddings = data["embeddings"]
```

Notes:

- `.npy` stores one array.
- `.npz` stores multiple arrays.
- `tolist()` converts a NumPy array back to a normal Python list.

For a beginner vector DB, `.npz` is a good first persistence format.

---

## 23. Measuring Search Speed

Your README says you want to measure search speed.

Use `time.perf_counter()`:

```python
import time

start = time.perf_counter()
results = db.search("Washington", k=2)
end = time.perf_counter()

print(f"Search took {end - start:.6f} seconds")
```

To compare pure Python and NumPy fairly:

1. Build embeddings once.
2. Reuse the same embeddings for both versions.
3. Time only the similarity search and sorting.
4. Do not include network calls to the embedding API in the benchmark.

Why?

The embedding API call will usually dominate runtime. If you include it in your
timing, you are mostly measuring network time, not NumPy.

For local benchmarking, use fake embeddings:

```python
import numpy as np

num_docs = 10_000
embedding_dim = 1536

embeddings = np.random.random((num_docs, embedding_dim)).astype(np.float32)
query = np.random.random(embedding_dim).astype(np.float32)
```

Then benchmark:

```python
start = time.perf_counter()
scores = embeddings @ query
top_indexes = np.argsort(scores)[::-1][:5]
end = time.perf_counter()

print(f"Search took {end - start:.6f} seconds")
```

This measures local numeric work only.

---

## 24. Measuring Memory Usage

Your README also says you want to measure memory usage.

Every NumPy array has `nbytes`:

```python
print(self.embeddings.nbytes)
```

That returns bytes.

Convert to megabytes:

```python
mb = self.embeddings.nbytes / (1024 * 1024)
print(f"{mb:.2f} MB")
```

Memory depends on:

```text
number_of_documents * embedding_dimension * bytes_per_number
```

For `float32`, each number uses 4 bytes.

For `float64`, each number uses 8 bytes.

Example:

```python
num_docs = 10_000
embedding_dim = 1536
bytes_per_float32 = 4

memory_bytes = num_docs * embedding_dim * bytes_per_float32
memory_mb = memory_bytes / (1024 * 1024)

print(memory_mb)
```

Output:

```text
58.59375
```

So 10,000 embeddings with 1536 dimensions stored as `float32` use about 58.6 MB
for the embedding matrix alone.

The text list uses additional memory.

---

## 25. A NumPy Version of Your Core Search

This is not a full replacement file. It is the core shape of what your next
implementation can look like.

```python
from openai import OpenAI
import numpy as np
import os


class VectorDB:
    def __init__(self, data: list[str]):
        self.texts = data
        self.embeddings: np.ndarray | None = None
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _generate_embeddings(self, text_to_embed: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text_to_embed,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding

    def build(self) -> None:
        all_embeddings = []

        for text in self.texts:
            embedding = self._generate_embeddings(text)
            all_embeddings.append(embedding)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        self.embeddings = self._normalize_rows(embeddings)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return matrix / norms

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def search(self, query_text: str, k: int) -> list[dict[str, float | str]]:
        if self.embeddings is None:
            raise ValueError("Call build() before search().")

        query_embedding = self._generate_embeddings(query_text)
        query = np.array(query_embedding, dtype=np.float32)
        query = self._normalize_vector(query)

        scores = self.embeddings @ query
        top_indexes = np.argsort(scores)[::-1][:k]

        matches = []
        for index in top_indexes:
            matches.append({
                "text": self.texts[index],
                "score": float(scores[index]),
            })

        return matches
```

Usage:

```python
DOCS = [
    "I am fine",
    "The capital of the United Kingdom is London",
    "Where is my money?",
]

db = VectorDB(DOCS)
db.build()

result = db.search("Washington", k=2)
print(result)
```

The main difference from your current version:

- Stored embeddings are generated once in `build`.
- Stored embeddings are kept in a 2D NumPy array.
- Stored embeddings are normalized once.
- Query embedding is normalized once per search.
- Search uses one matrix-vector multiplication: `self.embeddings @ query`.

---

## 26. Direct Mapping From Your Current Code to NumPy

### Current storage

```python
self.store = []
```

NumPy-oriented storage:

```python
self.texts = data
self.embeddings = None
```

### Current embedding storage

```python
temp_map = {}
embeddings = self._generate_embeddings(text)
temp_map["text"], temp_map["embeddings"] = text, embeddings
self.store.append(temp_map)
```

NumPy-oriented embedding storage:

```python
all_embeddings.append(embedding)
self.embeddings = np.array(all_embeddings, dtype=np.float32)
```

### Current magnitude

```python
self._magnitude(vector)
```

NumPy:

```python
np.linalg.norm(vector)
```

### Current dot product

```python
self._dot_product(vector_1, vector_2)
```

NumPy:

```python
np.dot(vector_1, vector_2)
```

or:

```python
vector_1 @ vector_2
```

### Current loop over stored embeddings

```python
for item in store_copy:
    embedding = item["embeddings"]
    cs_score = self._cosine_similarity(query_embeddings, embedding)
    item["score"] = cs_score
```

NumPy:

```python
scores = self.embeddings @ query
```

if both stored embeddings and query are already normalized.

### Current sorting

```python
store_copy.sort(key=_cosine_score, reverse=True)
```

NumPy:

```python
top_indexes = np.argsort(scores)[::-1][:k]
```

---

## 27. Common Beginner Mistakes

### Mistake 1: Using arrays with inconsistent embedding lengths

This is bad:

```python
np.array([
    [0.1, 0.2],
    [0.3, 0.4, 0.5],
])
```

Every embedding must have the same length.

Check:

```python
lengths = [len(embedding) for embedding in all_embeddings]
print(set(lengths))
```

You want one unique length.

### Mistake 2: Forgetting `axis=1`

This:

```python
np.linalg.norm(embeddings)
```

computes one norm for the entire matrix.

This:

```python
np.linalg.norm(embeddings, axis=1)
```

computes one norm per row.

For vector search, you usually want `axis=1`.

### Mistake 3: Losing the text-to-embedding relationship

If `texts` and `embeddings` get out of order, your search results will return
the wrong text.

This must always be true:

```python
texts[i]
```

belongs to:

```python
embeddings[i]
```

### Mistake 4: Sorting scores but not tracking indexes

This loses the relationship:

```python
sorted_scores = np.sort(scores)
```

Prefer:

```python
top_indexes = np.argsort(scores)[::-1][:k]
```

Indexes let you retrieve the matching text.

### Mistake 5: Including API calls in performance benchmarks

This measures network latency:

```python
db.search("query", k=5)
```

If `search` calls the embedding API, your benchmark includes the API request.

To measure NumPy search speed, benchmark only:

```python
scores = embeddings @ query
top_indexes = np.argsort(scores)[::-1][:k]
```

---

## 28. Practice Exercises

### Exercise 1: Convert a list to an array

Create a Python list:

```python
embedding = [0.1, 0.2, 0.3]
```

Convert it to a NumPy array with `dtype=np.float32`.

Print:

```python
array
array.shape
array.dtype
```

### Exercise 2: Reimplement `_magnitude`

Write:

```python
def magnitude(vector: np.ndarray) -> float:
    ...
```

Use `np.linalg.norm`.

Test it with:

```python
np.array([3, 4])
```

Expected result:

```text
5.0
```

### Exercise 3: Reimplement `_dot_product`

Write:

```python
def dot_product(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    ...
```

Use `np.dot` or `@`.

Test:

```python
dot_product(np.array([1, 2, 3]), np.array([4, 5, 6]))
```

Expected result:

```text
32
```

### Exercise 4: Compute cosine similarity for two vectors

Write:

```python
def cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    ...
```

Use:

```python
dot = vector_1 @ vector_2
norms = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
return dot / norms
```

### Exercise 5: Compute cosine similarity for many vectors

Create:

```python
embeddings = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
], dtype=np.float32)

query = np.array([1, 0, 0], dtype=np.float32)
```

Normalize the rows and query.

Compute:

```python
scores = embeddings @ query
```

Which row is most similar to the query?

### Exercise 6: Return top `k`

Given:

```python
texts = ["first", "second", "third"]
scores = np.array([0.3, 0.9, 0.5])
```

Return the top 2 matches as:

```python
[
    {"text": "second", "score": 0.9},
    {"text": "third", "score": 0.5},
]
```

Use `np.argsort`.

### Exercise 7: Add `build()` to your vector database

Move stored embedding generation out of `search`.

Target usage:

```python
db = VectorDB(DOCS)
db.build()
db.search("Washington", k=2)
```

This should prevent stored embeddings from being regenerated on every search.

---

## 29. Suggested Implementation Path for This Project

Follow this order:

1. Add NumPy to the project with `uv add numpy`.
2. Create a new implementation file, for example
   `src/numpy_implementation/pyvdb_numpy.py`.
3. Keep your existing pure Python implementation unchanged for comparison.
4. Add `self.texts` and `self.embeddings`.
5. Add a `build()` method that creates a 2D NumPy embedding matrix.
6. Add row normalization for stored embeddings.
7. In `search`, generate and normalize only the query embedding.
8. Compute scores with `self.embeddings @ query`.
9. Return top `k` with `np.argsort`.
10. Add timing tests that compare only local similarity search.
11. Add memory reporting with `self.embeddings.nbytes`.

That path matches your README:

```text
Level 0: pure Python lists
Level 0.5: NumPy arrays to store text-embedding pairs
```

---

## 30. The Core NumPy Ideas to Remember

For your vector database, these are the concepts that matter most:

```python
np.array(...)
```

Turns Python lists into NumPy arrays.

```python
array.shape
```

Tells you the structure of your data.

```python
array.dtype
```

Tells you the numeric type and memory cost.

```python
np.linalg.norm(vector)
```

Computes vector magnitude.

```python
np.linalg.norm(matrix, axis=1)
```

Computes one magnitude per row.

```python
vector_1 @ vector_2
```

Computes a dot product.

```python
matrix @ vector
```

Computes one dot product per matrix row.

```python
matrix / norms
```

Uses broadcasting to normalize many vectors.

```python
np.argsort(scores)[::-1][:k]
```

Gets indexes of the top `k` scores.

```python
array[indexes]
```

Selects rows or values by index.

If you understand those pieces, you can implement a simple brute force vector
database with NumPy.
