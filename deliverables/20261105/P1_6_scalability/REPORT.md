# P1.6 Scalability, release 20261105

## Time by population size, panel of 20 (seconds)

| population_variants | tree | adoption_gap | breadth | structural | coverage | greedy_adoption | greedy_breadth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.10135 | 8.9e-06 | 0.00047 | 2.4e-06 | 3.0e-06 | 0.00073 | 0.0872 |
| 2000 | 0.21112 | 8.7e-06 | 0.00068 | 2.4e-06 | 3.1e-06 | 0.00124 | 0.18425 |
| 4000 | 0.41627 | 8.7e-06 | 0.00101 | 2.3e-06 | 2.9e-06 | 0.00174 | 0.67095 |
| 8000 | 0.84093 | 8.9e-06 | 0.00162 | 2.4e-06 | 2.9e-06 | 0.00349 | 2.31374 |
| 16000 | 1.70344 | 8.9e-06 | 0.00305 | 2.4e-06 | 2.9e-06 | 0.00621 | 8.75395 |
| 32000 | 3.57091 | 8.8e-06 | 0.0054 | 2.4e-06 | 2.9e-06 | 0.01249 | 34.79433 |
| 64000 | 7.32732 | 9.0e-06 | 0.01089 | 2.6e-06 | 3.1e-06 | 0.02367 |  |
| 128000 | 15.01718 | 8.8e-06 | 0.01982 | 2.5e-06 | 3.0e-06 | 0.0505 |  |
| 256000 | 29.04865 | 9.1e-06 | 0.04203 | 2.5e-06 | 3.0e-06 | 0.13298 |  |

`adoption_gap`, `structural`, and `coverage` are per panel after one setup. `tree` also reports construction for all
475,340 edges in the largest case.

## Peak process memory, panel of 20 (MB)

Includes loading the foundation cache and pandas, which accounts for the floor in small populations.

| population_variants | tree | adoption_gap | breadth | structural | coverage | greedy_adoption | greedy_breadth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 154 | 154 | 154 | 154 | 154 | 154 | 154 |
| 2000 | 154 | 154 | 154 | 154 | 154 | 154 | 167 |
| 4000 | 154 | 154 | 154 | 154 | 154 | 154 | 307 |
| 8000 | 154 | 154 | 154 | 154 | 154 | 154 | 862 |
| 16000 | 158 | 156 | 156 | 157 | 156 | 157 | 3080 |
| 32000 | 206 | 204 | 205 | 212 | 206 | 206 | 11900 |
| 64000 | 310 | 310 | 311 | 311 | 312 | 311 |  |
| 128000 | 513 | 515 | 514 | 514 | 514 | 513 |  |
| 256000 | 865 | 864 | 867 | 866 | 867 | 866 |  |

## Time by panel size, real-scale population (seconds)

| panel_size | adoption_gap | breadth | structural | coverage | greedy_adoption | greedy_breadth |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 8.4e-06 | 0.0005 | 2.4e-06 | 2.5e-06 | 0.00083 | 1.06488 |
| 10 | 8.6e-06 | 0.00094 | 2.3e-06 | 2.7e-06 | 0.00162 | 1.49321 |
| 20 | 8.9e-06 | 0.00162 | 2.4e-06 | 2.9e-06 | 0.00349 | 2.31374 |
| 50 | 0.00001 | 0.00407 | 2.5e-06 | 3.9e-06 | 0.0081 | 5.18082 |
| 100 | 0.00001 | 0.0081 | 2.6e-06 | 5.9e-06 | 0.01618 | 9.08366 |

## Complexity

V is the number of variants, E the edges (about 1.8 V here), k the panel size, and C the candidate pool or structural cells.

| operation | time | memory |
| --- | --- | --- |
| tree | O(V log V) | O(V) |
| adoption_gap | O(k) | O(E) once, for the target branch masses |
| breadth | O(k V) | O(V) |
| structural | O(k + C) | O(C) |
| coverage | O(k) | O(V) |
| greedy_adoption | O(k V) | O(V + E) |
| greedy_breadth | O(k V C) with a stored kernel | O(V^2) kernel |

## Whole-pipeline stages on real data (seconds)

| stage | seconds |
| --- | --- |
| clean | 3933.3 |
| lineage | 1181.2 |
| validation | 769.3 |
| appendix_c | 155.5 |

## Notes

- Measurements are single-threaded (BLAS limited to one thread) on a shared 64-core machine, each in a fresh process.
- Populations above the real size replicate the real table under renamed families. They test computation, not realism.
- The tree builder here matches the validated geometry on the real population, to within 0.0e+00.
- The breadth greedy stores a full V by V kernel, which limits it to about 32,000 variants here (4 bytes times V squared, about
  4 GB). Since similarity depends only on the lowest common ancestor, it could use per-edge aggregation instead, but that
  has not been implemented. Larger cells are marked skipped in `scalability_grid.csv`.
