# CSE505 Project 8
## Metadata-Constrained Search in Vector Databases

This repository contains the implementation for our CSE505 Database Systems mini project.

## Research Question

How does metadata-filter selectivity affect the accuracy and efficiency of vector search under different execution strategies, and does the correlation between the filter predicate and query semantics change that relationship?

## Project Idea

Vector databases find documents that are similar in meaning to a query. In real applications, a query may also include metadata conditions such as category, year, language, or label.

This project compares four filtering strategies inside Chroma:

1. **Pre-filtering**: apply the filter first, then search.
2. **Post-filtering**: search first, then remove invalid results.
3. **Fixed over-fetching**: retrieve extra candidates, filter them, then keep the best valid results.
4. **Iterative over-fetching**: keep increasing the candidate size until enough valid results are found or a maximum limit is reached.

## Dataset

We use the **BanFakeNews** Bangla news dataset.

Main file:
`Authentic-48K.csv`

Optional file:
`Fake-1K.csv`

The dataset contains fields such as article ID, headline, content, date, category, and label.

The dataset is not included in this repository. Download it separately and update the dataset path in `cse505_project8.py`.

## Important: Dataset Path

Before running the code, update `DATA_DIR`.

For Windows, use a raw string:

```python
DATA_DIR = r"C:\Users\YOUR_NAME\Downloads\Project_DBMS"
```

For Linux or Google Colab:

```python
DATA_DIR = "/content/Project_DBMS"
```

## Embedding Model

We use `paraphrase-multilingual-MiniLM-L12-v2`.

It creates 384-dimensional embeddings and supports Bangla text.

## Vector Database

We use **Chroma** with an HNSW index and cosine distance.

Main index settings:

- M = 16
- construction_ef = 200
- search_ef = 100

## Queries

The experiment uses 20 Bangla queries from education, politics, technology, sports, and finance.

## Filter Selectivity

Selectivity means the percentage of documents that pass a filter.

We test approximately:

`100%, 50%, 20%, 10%, 5%, 1%, 0.5%, 0.1%`

We use two metadata conditions.

### Random / Uncorrelated
Each document gets a random bucket value.

### Structured / Correlated
Documents are ordered using the first PCA component of the embedding space, then bucket values are assigned from that order.

## Ground Truth

For every query and filter, the code calculates the exact filtered top-10 nearest neighbors using brute-force cosine similarity. This exact result is used as the ground truth.

## Evaluation Metrics

- Recall@10
- Completeness@10
- p50 latency
- p95 latency
- Candidate count
- GLS rho

## Controlled Ablation

The over-fetch multiplier is tested at:

`K'/K = 1, 2, 5, 10, 20, 50, 100`

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Update `DATA_DIR` in `cse505_project8.py`.

3. Run:

```bash
python cse505_project8.py
```

## Output Files

The script creates:

- `results.csv`
- `results_partial.csv`
- `ablation_kprime.csv`
- `summary_table.csv`
- `environment.json`
- `fig_recall.png`
- `fig_completeness.png`
- `fig_latency.png`
- `fig_ablation.png`

It may also create:

- `embeddings.npy`
- `chroma_store/`

## Main Findings

- Pre-filtering gives very high recall but is slow in our Chroma setup when many documents pass the filter.
- Post-filtering is fast but performs poorly at low selectivity.
- Fixed over-fetching improves recall, but one fixed multiplier does not work well for every selectivity level.
- Iterative over-fetching gives a better balance because it increases the candidate size only when needed.
- Selectivity alone does not fully explain search difficulty.
- The local relationship between the query and metadata distribution also affects recall.

## Reproducibility

Main settings:

- Seed = 42
- K = 10
- Timed repeats = 3
- Embedding batch size = 256
- Insert batch size = 2000
- Iterative maximum K' = 2000

Software versions are stored in `environment.json`.

## Suggested Repository Structure

```text
CSE505-Metadata-Constrained-Vector-Search/
├── README.md
├── cse505_project8.py
├── requirements.txt
├── environment.json
├── results/
│   ├── results.csv
│   ├── results_partial.csv
│   ├── summary_table.csv
│   └── ablation_kprime.csv
├── figures/
│   ├── fig_recall.png
│   ├── fig_completeness.png
│   ├── fig_latency.png
│   └── fig_ablation.png
└── report/
    └── final_report.pdf
```

## Notes

- Do not upload `embeddings.npy` unless needed because it can be large.
- Do not upload `chroma_store/` unless needed.
- Do not upload the dataset unless its license allows redistribution.
- Absolute latency can change on different hardware. The comparison between strategies is more important.

## Course

**CSE505: Database Systems**

Mini Project 8: **Metadata-Constrained Search in Vector Databases**
