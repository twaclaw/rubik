## Kociemba's two-Phase algorithm for solving Rubik's cube

This implementation is based on the [Python implementation](https://github.com/hkociemba/RubiksCube-TwophaseSolver.git) by Kociemba.

I decided to rewrite the code as a learning exercise to better understand the mechanics of the algorithm. My guess is that the original code was written primarily for pedagogical purposes, to be as clear as possible.

In rewriting the code, I included some 'optimizations' here and there.

Here is a list of the changes I made:

- I kept the same naming conventions and even the original comments, which are very helpful.
- I did some non-funcional refactoring, structuring the original functional code into classes.
- Then, to avoid circular imports, I fused `enums.py`, `cubie.py` and `symmetries.py` from the original code into a single [cubie.py](./cubie.py) file.
- The most significant changes are in the `CubieClass` in [cubie.py](./cubie.py):
  - I slighly changed the way permutations and rotations are represented, grouping them into a single `Numpy` array. I made this change quite early on, anticipating it would be useful later on, but it is rather inconsequential.
  - The main change is that I vectorized all the coordinate functions.
- While studying the two-phase algorithm and related literature I discovered the concept of Lehmer codes. I found the idea so interesting, that I decided to create an independent library, [lehmer](https://github.com/twaclaw/lehmer), a vectorized, batchable library for computing Lehmer codes. Kociemba's code does something similar &mdash;but not exactly the same&mdash; to computing a Lehmer code of a permutation. Instead of counting the number of inversions, his code counts the number of required rotations. This was my riskiest change because it implies that the generated tables differ from the original ones. However, I bet that, as long as consistency was maintained, it wouldn't matter, and I was right.
- Instead of `array`, the tables used Numpy's binary format with [memory mapping](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) to reduce loading time.
- The original code launches multiple threads to find shorter solutions. I didn't implement that because I was interested in the mechanics of the algorithm. However, I kept something similar to the original in [solver2.py](./solver2.py).
- Finally, because I was already using [Rich](https://github.com/Textualize/rich) for plotting the cubes in the console, I used it to beautify the output (e.g., progress bars).

The original code is quite efficient. It takes about ten minutes to generate all the tables, and obtaining a solution takes just a few seconds (on a MacBook Air with an M2 chipset and 24 GB of RAM). My implementation takes less than five minutes to generate the tables and is generally twice as fast at obtaining solutions.

### To do / possible improvements

My original focus was on vectorizing the coordinate functions. I have not finished that part yet; a few Python loops could still be eliminated.

Another potential optimization would be to leverage the fact that the Lehmer codes library I implemented is batchable. This would allow multiple non-sequential coordinates to be put in RAM, batched, and processed together.

In general, because most of the operations use small Numpy arrays, the benefits of the vectorization are limited. If it is possible to group some of the operations in larger or multidimensional arrays, that could lead to further speed improvements.

### References

- [Kociemba's home page](https://kociemba.org/)
- [Kociemba's GitHub](https://github.com/hkociemba)
- [My library to compute Lehmer codes](https://github.com/Textualize/rich)
