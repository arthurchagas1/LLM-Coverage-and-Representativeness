# RQ1 Coverage of existing panels, release 20261105

Historical audit: each of the 46 scoreable panels against the variants available by its cutoff.

## Distribution of the coordinates over papers

| coordinate | median | q25 | q75 | min | max |
| --- | --- | --- | --- | --- | --- |
| Panel size | 7.5 | 4 | 10 | 1 | 51 |
| Distinct families | 4 | 2 | 5 | 1 | 9 |
| Largest family share | 0.4817 | 0.3333 | 0.7292 | 0.2143 | 1 |
| Adoption gap G_a | 1.8201 | 1.7184 | 1.8848 | 1.4152 | 1.9954 |
| Lineage breadth B | 0.0055 | 0.0054 | 0.0066 | 0.0054 | 0.0219 |
| Structural imbalance D_inf | 0.6705 | 0.4997 | 0.8327 | 0.3101 | 1 |
| C(0) | 0.0019 | 0.001 | 0.005 | 4.8e-05 | 0.013 |
| C(1) | 0.0301 | 0.0069 | 0.0472 | 1.8e-05 | 0.191 |

## Percentiles against same-size references

Share of the 120 reference panels that the observed panel matches or beats (larger is better).

| reference | coordinate | median | q25 | q75 | papers_below_0.05 | papers_below_0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| Adoption-weighted | adoption | 0 | 0 | 0.015 | 41 | 45 |
| Adoption-weighted | breadth | 0.067 | 0 | 0.281 | 19 | 39 |
| Adoption-weighted | structural | 0.029 | 0 | 0.279 | 25 | 37 |
| Family-stratified | adoption | 0 | 0 | 0.008 | 37 | 44 |
| Family-stratified | breadth | 0 | 0 | 0.062 | 33 | 43 |
| Family-stratified | structural | 0.017 | 0 | 0.15 | 28 | 40 |
| Uniform | adoption | 0.429 | 0.106 | 0.852 | 7 | 27 |
| Uniform | breadth | 0.008 | 0 | 0.056 | 33 | 44 |
| Uniform | structural | 0.008 | 0 | 0.158 | 29 | 41 |

## Where the structural imbalance comes from

| observed_structural_partition | papers |
| --- | --- |
| family | 12 |
| stage | 11 |
| size | 8 |
| role | 8 |
| series | 6 |
| base | 1 |

Most frequent cells:

| observed_structural_partition | observed_structural_cell | papers |
| --- | --- | --- |
| stage | instruction_or_chat_tuned | 10 |
| family | llama | 9 |
| role | instruct | 8 |
| series | deepseek / deepseek-r1 | 2 |
| size | 13B-34B | 2 |
| size | 7B-13B | 2 |
| series | qwen / qwen2.5 | 2 |
| series | llama / llama3.1 | 2 |
| size | unknown | 2 |
| base | qwen / qwen3 / base::qwen::qwen3::235b | 1 |

In 42 of 46 papers the worst cell is over-represented by the panel (positive signed error).

## Panel mass by family against target mass, averaged over papers

`panel` is the mean share of panel mass, `adoption` and `breadth` the mean share of each target at the paper's cutoff.

| family | panel | adoption | breadth | papers_with_family |
| --- | --- | --- | --- | --- |
| llama | 0.396 | 0.108 | 0.001 | 39 |
| qwen | 0.173 | 0.159 | 8.0e-04 | 27 |
| deepseek | 0.125 | 0.005 | 7.2e-04 | 19 |
| mistral | 0.092 | 0.041 | 9.6e-04 | 22 |
| gemma | 0.034 | 0.006 | 4.5e-04 | 11 |
| glm | 0.031 | 0.005 | 0.002 | 9 |
| yi | 0.017 | 0.014 | 0.002 | 7 |
| baichuan | 0.016 | 0.001 | 0.003 | 4 |
| all other families | 0.116 | 0.66 | 0.99 |  |

Families with the most adoption mass:

| family | panel | adoption | breadth | papers_with_family |
| --- | --- | --- | --- | --- |
| gpt2 | 0 | 0.211 | 0.002 | 0 |
| t5 | 0.012 | 0.203 | 0.003 | 3 |
| qwen | 0.173 | 0.159 | 8.0e-04 | 27 |
| llama | 0.396 | 0.108 | 0.001 | 39 |
| gpt-neox | 0 | 0.043 | 0.003 | 0 |
| mistral | 0.092 | 0.041 | 9.6e-04 | 22 |
| opt | 0 | 0.032 | 0.003 | 0 |
| marian | 0 | 0.032 | 0.002 | 0 |

## By source

| source | papers | median_n | median_gap | median_uniform_gap_percentile | median_uniform_breadth_percentile | median_structural |
| --- | --- | --- | --- | --- | --- | --- |
| ACL Anthology | 19 | 7 | 1.868 | 0.267 | 0.008 | 0.777 |
| ML conferences | 17 | 9 | 1.763 | 0.767 | 0.008 | 0.598 |
| Not recorded | 1 | 5 | 1.885 | 0.267 | 0.008 | 0.798 |
| arXiv | 9 | 7 | 1.754 | 0.633 | 0.058 | 0.814 |
