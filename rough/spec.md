### python implemenation

Util functions

- tokenize
input/args: text: str
output/return: list[str]
use python re library, compile() method to find a pattern
convert all patterns to lowercase

- stable_hash
input/args: text: str
output/return: int
create a digest using hashlib's blake2b()
return an int hash from the bytes of created digest

- dot
input/args: left: list[float], right: list[float]
output/return: float
    - operations
    raise ValueError() if vector length mismatch
    zip left and right vectors, multiply and add all l[i] and r[i]
    return the total

- l2_norm
input/args: vector: list[float]
output/returns: float
    call dot() on input vector * vector
    apply python math.sqrt() on the value on op 1
    return value of op 2

- normalize
input/args: vector: list[float]
output/returns: list[float]
    operations
    - call l2_norm() on input vector
    - raise ValueError if l2 of input vector is 0
    - perform 1 / l2_norm of input vector, inverse norm
    - perform list comprehension multiplying all points/values in input vector with inverse norm
    return last operation

- hashed_embedding
input/args: text: str, dimensions: int
output/return: list[float] (list of floating point values)
    operations
    - create a list (vector) with size of dimensions
    - call tokenize on input text, will return a list of tokens from input text
    - loop through tokens list and for each token in tokens list, perform the following:
        - create a bucket for the token by calling stable_hash() on "bucket" + {token}, take modulo of hash / dimensions
        - create a sign for each token by:
            - create a stable_hash of "sign" + {token}
            - perform bitwise comparison of stable_hash by & 1, if result of comparison 
            1, set sign to -1.0, else (ie, &1 is 0) set sign to 1.0
            assign sign value to bucket index in vector list
    - outside the loop, call normalize() on vector
    return normalize(vector)

- metadata_matches
input/args: metadata: dict[str, Any], filters: dict[str, Any] | None
output/return: bool
    - operations: you are comparing two dictionaries, if the value in metadata != value in filters, return False, else, return True, so you'll be returning all records
    handle, base case of no filters, return True
    loop through key, value in filters, 
        set up the filter condition, if metadata.get(key), which returns value of key in metadata, is not equal to value (in filters)
        return False
    leave the loop and return True

create the SearchResult class
call the dataclass decorator
class properties:
- record_id: str, a unique identifier for each returned matching record?
- score: float, the similarity score between query and matching record
- payload: str, the text stored
- metadata: dict[str, Any], probably contains all properties, with record_id as key, and list containing all other properties as values

class implementation of the PythonVectorDB
data layout: ids, payloads, metadata and vectors stored in parallel lists
id_to_index maps each record's unique id to positions across parallel lists.
so for instance, '001' would always be list[1] across, ids, payloads, metadata and vector lists

- init_function()
input/args: self, dimensions: int
returns None
    - operations
    1. raise value error if dimensions is negative
    2. assign self.dimensions to dimensions value passed in during object initialization
    3. create parallel lists: ids <list[str]>, payloads <list[str]>, metadata <list[dict[str, Any]]>, vectors <list[list[float]]>, and id to index hash map <dict[str, Any]>

special (dunder) methods:
__len__()
the length of the vectordb is the length of the ids list
return that

class methods
- validate:
implement as a hidden method, meaning that it should not be invoked directly through an object's dot notation. it's meant to be used internally in the class
input/args: self, vector: list[float]
output/return: list[float]
operations:
    - 1. check if the length of the input vector is not equal to the dimensions value added to the db object at initialization. if it is not the same, raise value error.
    - 2. create a copy of the input vector, by performing a list comprehension where you convert every value in the vector to a float
    3. call normalize() on the copy and return it

- upsert
this is how you add a new record to the database
input/args: self, record_id <str>, vector <list[float]>, payload <str>, metadata dict<str, Any or None>, initialize metadat to None
output/return: None
operations:
    1. normalize the input vector using the hidden validate_vector()
    2. create a record metadata variable to hold a shallow dict copy of metadata or an empty dict
    3. create an existing index variable with record_id as the key and id_to_index as the dict. the value of existind index will be the value of the key of id_to_index
    4. if existing index is not empty, ie not None, do the following:
        a. add normalized to vectors list at index existing_index
        b. add payload to payloads list at index existing_index
        c. add record metadata to metadata list at index existing_index
        return
    5. outside the if statement, ie, in the case where this is a new record, do the following:
        a. add record_id as a key to the id_to_index dict and its value should be set to the length of self.ids, so, effectively place it at the last position in the index
        b. add record_id to the ids list
        c. add payload to the payloads list
        d. add record_metadata to the metadata list
        e. add normalized to the vectors list

- search
input/args: self, query_vector <list[float]>, k: <int, default value of 5>, filters <dict[str, Any] or None>
output/return: list[SearchResult]
operations
    1. if k is less than or equal to 0, return an empty list
    2. create a candidates variable with a type list[SearchResult] initialized to an empty list
    3. create a normalized variable to hold the result of passing 





