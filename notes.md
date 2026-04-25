
Why pre-normalizing the embeddings works:

A unit vector is a vector whose total length is exactly 1, not a vector whose individual entries are all 1. For a vector like `[3, 4]`, the length is found using the Euclidean norm, meaning the straight-line distance from the origin: `sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5`. To normalize it, you divide each component by the original length, giving `[3/5, 4/5] = [0.6, 0.8]`. The reason `[0.6, 0.8]` has length 1 is that `sqrt(0.6² + 0.8²) = sqrt(0.36 + 0.64) = sqrt(1) = 1`. So normalization keeps the direction of the vector the same but rescales its size so its length becomes exactly 1.


