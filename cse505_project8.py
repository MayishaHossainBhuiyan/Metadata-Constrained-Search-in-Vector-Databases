"""
CSE505 Mini Project 8 - Metadata-Constrained Search in Vector Databases
======================================================================

Our Research question
-----------------
How does metadata-filter selectivity affect the accuracy and efficiency of vector
search under different execution strategies, and does the correlation between the
filter predicate and query semantics change that relationship?


Outputs: results.csv, ablation_kprime.csv, summary_table.csv, environment.json,
         fig_recall.png, fig_completeness.png, fig_latency.png, fig_ablation.png
"""

# %% ===================================================================== CONFIG

import os

# --- dataset paths -----------------------------------------------------------
DATA_DIR = "C:\Users\88016\Downloads\Project_DBMS"
AUTHENTIC_CSV = os.path.join(DATA_DIR, "Authentic-48K.csv")

# Loading only the authentic file leaves `label` constant at 1, which makes it
# useless as a filter attribute. If you have the fake-news half, put it in the
# same directory and `label` becomes a real binary predicate. Set to None to skip.
FAKE_CSV = os.path.join(DATA_DIR, "Fake-1K.csv")

MOUNT_DRIVE = False         # True only on Colab
PERSIST_DIR = "./chroma_store"

# All figures, CSVs and logs land here.
OUTPUT_DIR = os.path.join(DATA_DIR, "project figure")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    """Resolve an output filename inside OUTPUT_DIR."""
    return os.path.join(OUTPUT_DIR, filename)

# --- experiment parameters ---------------------------------------------------
SEED = 42
K = 10                      # neighbours requested per query
REPEATS = 3                 # timed repeats per measurement (after warm-up)
EMBED_BATCH = 256
INSERT_BATCH = 2000
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# bucket thresholds out of 1000 -> selectivity 100% ... 0.1%
SELECTIVITIES = [1000, 500, 200, 100, 50, 10, 5, 1]
ABLATION_THRESHOLDS = [500, 100, 50, 10, 5]
ABLATION_MULTS = [1, 2, 5, 10, 20, 50, 100]
ITERATIVE_CAP = 2000        # max K' before the iterative strategy gives up

INDEX_CONFIG = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:M": 16,
    "hnsw:search_ef": 100,
}


# %% ========================================================= 0. DEPENDENCIES

import importlib
import subprocess
import sys

# Colab ships pandas/numpy/sklearn/matplotlib but not these two.
REQUIRED = {
    "chromadb": "chromadb",
    "sentence_transformers": "sentence-transformers",
}


def ensure_dependencies():
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Installing:", " ".join(missing))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing]
        )
        importlib.invalidate_caches()
        print("Done. If imports still fail in Colab, restart the runtime and re-run.")
    else:
        print("All dependencies present.")


ensure_dependencies()


# %% ============================================================ 1. ENVIRONMENT

import json
import random
import shutil
import time
import platform

import numpy as np
import pandas as pd

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:                                    # not in Colab
        print("Drive mount skipped:", exc)


def record_environment():
    """Version capture - the report needs this for the reproducibility section."""
    import sentence_transformers
    import chromadb
    import sklearn

    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "chromadb": chromadb.__version__,
        "sklearn": sklearn.__version__,
        "model": MODEL_NAME,
        "seed": SEED,
        "K": K,
    }
    with open(out("environment.json"), "w") as handle:
        json.dump(env, handle, indent=2)
    print(json.dumps(env, indent=2))
    return env


# %% ================================================================ 2. DATASET

def load_dataset():
    """Load the authentic CSV, plus the fake CSV when available."""
    frames = []

    part = pd.read_csv(AUTHENTIC_CSV)
    print(f"{AUTHENTIC_CSV}: {part.shape}")
    frames.append(part)

    if FAKE_CSV and os.path.exists(FAKE_CSV):
        part = pd.read_csv(FAKE_CSV)
        print(f"{FAKE_CSV}: {part.shape}")
        frames.append(part)
    else:
        print(f"\n!! {FAKE_CSV} not found - `label` will be constant and unusable "
              f"as a filter.\n   Only `category`, `year` and the synthetic buckets "
              f"will be available.\n   Note this as a limitation in the report.\n")

    frame = pd.concat(frames, ignore_index=True)
    print("Combined:", frame.shape)
    return frame


def prepare_metadata(frame):
    """Build the metadata table, including the two synthetic selectivity columns.

    `bucket`      - uniform random in [0,1000). `bucket < B` gives a filter of
                    selectivity exactly B/1000, uncorrelated with the vectors.
    `corr_bucket` - assigned by rank along PC1 of the embedding space (filled in
                    later, once embeddings exist), so a filter selects a
                    semantically contiguous slab at the same selectivity.

    Comparing the two at matched selectivity is what isolates the correlation
    effect - the axis the literature (Li et al. O5, GLS) flags as open.
    """
    frame["text"] = frame["headline"].fillna("") + " " + frame["content"].fillna("")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["year"] = frame["date"].dt.year

    meta = frame[["articleID", "text", "category", "year", "label"]].copy()
    meta = meta.dropna(subset=["text", "category"]).reset_index(drop=True)
    meta["year"] = meta["year"].fillna(-1).astype(int)
    meta["doc_id"] = np.arange(len(meta))       # contiguous == embedding row index

    rng = np.random.default_rng(SEED)
    meta["bucket"] = rng.integers(0, 1000, size=len(meta))
    meta["corr_bucket"] = -1

    print("\nShape:", meta.shape)
    print("\nCategory selectivity (%):")
    print((meta["category"].value_counts(normalize=True) * 100).round(2))
    print("\nYear:", dict(meta["year"].value_counts()))
    print("Label:", dict(meta["label"].value_counts()))
    return meta


# %% ============================================================== 3. EMBEDDING

def build_embeddings(meta):
    """Encode all documents. Unit-normalised so dot product == cosine, which keeps
    the brute-force ground truth exactly consistent with Chroma's cosine space."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    # Reuse cached embeddings if they match the corpus - makes restarts cheap.
    if os.path.exists("embeddings.npy"):
        cached = np.load("embeddings.npy")
        if len(cached) == len(meta):
            print(f"Reusing cached embeddings.npy {cached.shape}")
            return model, cached.astype("float32"), 0.0
        print("Cached embeddings do not match corpus size - re-encoding.")

    start = time.perf_counter()
    embeddings = model.encode(
        meta["text"].tolist(),
        batch_size=EMBED_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    elapsed = time.perf_counter() - start

    np.save("embeddings.npy", embeddings)
    print(f"\nShape: {embeddings.shape} | {elapsed:.1f}s")
    print("Norm check (expect ~1.0):", float(np.linalg.norm(embeddings[0])))
    return model, embeddings, elapsed


def add_correlated_bucket(meta, embeddings):
    """Rank documents along PC1 and assign corr_bucket by that rank."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=1, random_state=SEED)
    projection = pca.fit_transform(embeddings).ravel()

    order = np.argsort(projection)
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(len(order))
    meta["corr_bucket"] = (ranks * 1000 // len(meta)).astype(int)

    # Sanity check: the correlated slab should be more homogeneous than a random one.
    corr = meta.loc[meta["corr_bucket"] < 100, "doc_id"].values[:300]
    rand = meta.loc[meta["bucket"] < 100, "doc_id"].values[:300]
    print("PC1 explained variance:", round(float(pca.explained_variance_ratio_[0]), 4))
    print("Mean pairwise sim - correlated slab:",
          round(float(np.mean(embeddings[corr] @ embeddings[corr].T)), 4))
    print("Mean pairwise sim - random slab:    ",
          round(float(np.mean(embeddings[rand] @ embeddings[rand].T)), 4))
    return meta


# %% ================================================================= 4. INDEX

def build_index(meta, embeddings):
    """Create the Chroma collection and record build time and store size."""
    import chromadb
    from chromadb.config import Settings
    from tqdm import tqdm

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    client = chromadb.PersistentClient(
        path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False)
    )

    config = dict(INDEX_CONFIG)
    try:
        collection = client.create_collection(name="banfakenews", metadata=config)
    except Exception as exc:
        print("Full HNSW config rejected, falling back to space only:", exc)
        config = {"hnsw:space": "cosine"}
        collection = client.create_collection(name="banfakenews", metadata=config)

    start = time.perf_counter()
    for begin in tqdm(range(0, len(meta), INSERT_BATCH), desc="insert"):
        end = min(begin + INSERT_BATCH, len(meta))
        chunk = meta.iloc[begin:end]
        collection.add(
            ids=chunk["doc_id"].astype(str).tolist(),
            documents=chunk["text"].tolist(),
            embeddings=embeddings[begin:end].tolist(),
            metadatas=[
                {
                    "category": str(row.category),
                    "year": int(row.year),
                    "label": int(row.label),
                    "bucket": int(row.bucket),
                    "corr_bucket": int(row.corr_bucket),
                }
                for row in chunk.itertuples()
            ],
        )
    build_time = time.perf_counter() - start

    size_mb = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(PERSIST_DIR)
        for name in names
    ) / 1e6

    print(f"Count: {collection.count()} | build {build_time:.1f}s | {size_mb:.1f} MB")
    print("Config:", config)
    return collection, build_time, size_mb, config


# %% ========================================== 5. GROUND TRUTH AND GLS METRIC

QUERIES = [
    "বিশ্ববিদ্যালয় ভর্তি", "শিক্ষা নীতি পরিবর্তন", "স্কুল শিক্ষার্থী", "পরীক্ষার ফলাফল",
    "নির্বাচন কমিশন", "রাজনৈতিক দল", "সংসদ নির্বাচন", "মন্ত্রিসভার সিদ্ধান্ত",
    "কৃত্রিম বুদ্ধিমত্তা", "তথ্য প্রযুক্তি", "মোবাইল ইন্টারনেট", "সাইবার নিরাপত্তা",
    "বাংলাদেশ ক্রিকেট দল", "ফুটবল ম্যাচ", "বিশ্বকাপ ক্রিকেট", "খেলোয়াড় নির্বাচন",
    "শেয়ার বাজার", "ব্যাংক ঋণ", "অর্থনৈতিক প্রবৃদ্ধি", "বিনিয়োগ বাজার",
]


def build_mask(meta, field, threshold):
    """Boolean mask over meta rows for the predicate `field < threshold`."""
    return meta[field].values < threshold


def exact_topk(embeddings, query_vec, mask, k):
    """Exact filtered top-k by brute force. This is the ground truth that makes
    recall meaningful - the original notebook had no ground truth at all, so its
    'recall' could only ever measure how many result slots got filled."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.array([], dtype=int), np.array([])
    sims = embeddings[idx] @ query_vec
    if len(idx) <= k:
        order = np.argsort(-sims)
        return idx[order], sims[order]
    part = np.argpartition(-sims, k)[:k]
    part = part[np.argsort(-sims[part])]
    return idx[part], sims[part]


def gls_rho(embeddings, query_vec, mask, k=100):
    """Global-Local Selectivity correlation (Amanbayev et al., 2026).

    sigma_g = fraction of the corpus passing the filter
    sigma_l = fraction of the query's unfiltered top-k passing the filter
    rho = (r-1)/(r+1) where r = sigma_l/sigma_g

    rho > 0 means the filter enriches the query neighbourhood, rho < 0 depletes it.
    """
    sigma_g = float(mask.mean())
    if sigma_g == 0:
        return float("nan")
    sims = embeddings @ query_vec
    top = np.argpartition(-sims, k)[:k]
    sigma_l = float(mask[top].mean())
    if sigma_l == 0:
        return -1.0
    ratio = sigma_l / sigma_g
    return (ratio - 1.0) / (ratio + 1.0)


# %% ============================================================ 6. STRATEGIES

def make_strategies(collection, meta):
    """Four execution strategies, each returning (retrieved_ids, candidates_examined).

    Filtering is resolved against in-memory numpy arrays keyed by doc_id rather than
    by asking Chroma to return metadata payloads. Chroma stores metadata in SQLite,
    so `include=["metadatas"]` deserialises thousands of dicts per query - on a large
    store this dominated runtime. The predicate semantics are identical; only the
    lookup path changed.
    """
    attr = {field: meta[field].values for field in ("bucket", "corr_bucket")}
    total = collection.count()          # fetched once, not per query

    def where_clause(field, threshold):
        return {field: {"$lt": int(threshold)}}

    def pre_filter(query_vec, field, threshold, k):
        """Native filtered search - Chroma applies the predicate during retrieval."""
        res = collection.query(
            query_embeddings=[query_vec.tolist()], n_results=k,
            where=where_clause(field, threshold), include=[],
        )
        ids = [int(i) for i in res["ids"][0]]
        return ids, len(ids)

    def post_filter(query_vec, field, threshold, k):
        """Fetch exactly k unfiltered, then filter. The honest post-filter baseline -
        the original used fetch_k=20 for top_k=5, already 4x over-fetching."""
        res = collection.query(
            query_embeddings=[query_vec.tolist()], n_results=k, include=[],
        )
        ids = [int(i) for i in res["ids"][0]]
        keep = [i for i in ids if attr[field][i] < threshold]
        return keep[:k], len(ids)

    def over_fetch(query_vec, field, threshold, k, mult=10):
        """Fetch k*mult unfiltered, filter, truncate to k."""
        k_prime = min(k * mult, total)
        res = collection.query(
            query_embeddings=[query_vec.tolist()], n_results=k_prime, include=[],
        )
        ids = [int(i) for i in res["ids"][0]]
        keep = [i for i in ids if attr[field][i] < threshold]
        return keep[:k], k_prime

    def iterative(query_vec, field, threshold, k, cap=ITERATIVE_CAP):
        """Double K' until k results are found or the cap is hit (pgvector style)."""
        k_prime, examined, keep = k, 0, []
        while k_prime <= cap:
            res = collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=int(min(k_prime, total)), include=[],
            )
            ids = [int(i) for i in res["ids"][0]]
            examined = len(ids)
            keep = [i for i in ids if attr[field][i] < threshold]
            if len(keep) >= k or examined < k_prime:
                return keep[:k], examined
            k_prime *= 2
        return keep[:k], examined

    return {
        "pre_filter": pre_filter,
        "post_filter": post_filter,
        "over_fetch": over_fetch,
        "iterative": iterative,
    }, over_fetch


# %% ====================================================== 7. MAIN EXPERIMENT

MODES = {"uncorrelated": "bucket", "correlated": "corr_bucket"}


def run_sweep(meta, embeddings, query_emb, strategies):
    """Selectivity sweep - the controlled ablation the spec requires and the axis
    the research question is about. The original notebook never varied it."""
    from tqdm import tqdm

    rows = []
    for mode_name, field in MODES.items():
        for threshold in tqdm(SELECTIVITIES, desc=mode_name):
            mask = build_mask(meta, field, threshold)
            if mask.sum() < K:
                continue
            sigma_g = float(mask.mean())

            for q_idx, query_vec in enumerate(query_emb):
                truth = set(exact_topk(embeddings, query_vec, mask, K)[0].tolist())
                rho = float(gls_rho(embeddings, query_vec, mask))

                for name, fn in strategies.items():
                    fn(query_vec, field, threshold, K)          # warm-up, discarded
                    lats = []
                    for _ in range(REPEATS):
                        start = time.perf_counter()
                        got, candidates = fn(query_vec, field, threshold, K)
                        lats.append((time.perf_counter() - start) * 1000)

                    rows.append({
                        "mode": mode_name,
                        "selectivity": sigma_g,
                        "threshold": threshold,
                        "query_idx": q_idx,
                        "query": QUERIES[q_idx],
                        "strategy": name,
                        "recall_at_k": len(set(got) & truth) / max(len(truth), 1),
                        "completeness_at_k": len(got) / K,
                        "n_returned": len(got),
                        "candidates": candidates,
                        "gls_rho": rho,
                        "lat_p50_ms": float(np.percentile(lats, 50)),
                        "lat_p95_ms": float(np.percentile(lats, 95)),
                    })

            pd.DataFrame(rows).to_csv(out("results_partial.csv"), index=False)

    results = pd.DataFrame(rows)
    results.to_csv(out("results.csv"), index=False)
    print("\nRuns:", len(results))
    print(results.groupby(["mode", "strategy"])[
        ["recall_at_k", "completeness_at_k", "lat_p50_ms"]].mean().round(3))
    return results


def run_ablation(meta, embeddings, query_emb, over_fetch):
    """Sweep the over-fetch multiplier K'. Isolates one design choice and shows the
    recall/latency price of over-fetching as selectivity drops."""
    rows = []
    for threshold in ABLATION_THRESHOLDS:
        mask = build_mask(meta, "bucket", threshold)
        if mask.sum() < K:
            continue
        sigma_g = float(mask.mean())

        for q_idx, query_vec in enumerate(query_emb):
            truth = set(exact_topk(embeddings, query_vec, mask, K)[0].tolist())
            for mult in ABLATION_MULTS:
                over_fetch(query_vec, "bucket", threshold, K, mult=mult)   # warm-up
                lats = []
                for _ in range(3):
                    start = time.perf_counter()
                    got, candidates = over_fetch(
                        query_vec, "bucket", threshold, K, mult=mult)
                    lats.append((time.perf_counter() - start) * 1000)

                rows.append({
                    "selectivity": sigma_g,
                    "mult": mult,
                    "query_idx": q_idx,
                    "recall_at_k": len(set(got) & truth) / max(len(truth), 1),
                    "completeness_at_k": len(got) / K,
                    "candidates": candidates,
                    "lat_p50_ms": float(np.percentile(lats, 50)),
                })

    ablation = pd.DataFrame(rows)
    ablation.to_csv(out("ablation_kprime.csv"), index=False)
    print(ablation.groupby(["selectivity", "mult"])[
        ["recall_at_k", "lat_p50_ms"]].mean().round(3).head(30))
    return ablation


# %% ================================================================ 8. FIGURES

def make_figures(results, ablation):
    """All figures read from the dataframes. The original hardcoded
    `recalls = [1.00, 0.82, 0.91]`, so re-running left stale numbers in the charts."""
    import matplotlib
    matplotlib.use("Agg")           # headless server - write PNGs, never block
    import matplotlib.pyplot as plt

    def by_selectivity(column, ylabel, filename):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for axis, mode in zip(axes, ["uncorrelated", "correlated"]):
            subset = results[results["mode"] == mode]
            if subset.empty:
                continue
            for strategy in sorted(subset["strategy"].unique()):
                grouped = (subset[subset["strategy"] == strategy]
                           .groupby("selectivity")[column].mean().sort_index())
                axis.plot(grouped.index * 100, grouped.values, marker="o", label=strategy)
            axis.set_xscale("log")
            axis.set_xlabel("Filter selectivity (% passing)")
            axis.set_title(mode)
            axis.grid(alpha=0.3)
        axes[0].set_ylabel(ylabel)
        axes[0].legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        pass  # savefig only - no interactive display on server

    by_selectivity("recall_at_k", f"Recall@{K}", out("fig_recall.png"))
    by_selectivity("completeness_at_k", f"Completeness@{K}", out("fig_completeness.png"))
    by_selectivity("lat_p50_ms", "p50 latency (ms)", out("fig_latency.png"))

    plt.figure(figsize=(7, 4.2))
    for sel in sorted(ablation["selectivity"].unique()):
        grouped = (ablation[ablation["selectivity"] == sel]
                   .groupby("mult")["recall_at_k"].mean())
        plt.plot(grouped.index, grouped.values, marker="s", label=f"{sel * 100:.1f}%")
    plt.xscale("log")
    plt.xlabel("Over-fetch multiplier K'/K")
    plt.ylabel(f"Recall@{K}")
    plt.title("Over-fetch ablation")
    plt.legend(title="selectivity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out("fig_ablation.png"), dpi=150)
    # savefig only - no interactive display on server

    print("\nGLS by mode:")
    print(results.groupby("mode")["gls_rho"].describe().round(3))


# %% ========================================================= 9. ERROR ANALYSIS

def error_analysis(results, meta, embeddings, query_emb, strategies, n=5):
    """Qualitative error analysis - required by the spec. Prints what each failing
    strategy returned versus what it should have returned."""
    failures = (results[(results["strategy"] != "pre_filter")
                        & (results["recall_at_k"] < 0.5)]
                .sort_values("recall_at_k").head(n))

    for _, row in failures.iterrows():
        field = MODES[row["mode"]]
        mask = build_mask(meta, field, int(row["threshold"]))
        query_vec = query_emb[int(row["query_idx"])]
        truth_ids, truth_scores = exact_topk(embeddings, query_vec, mask, K)
        got, _ = strategies[row["strategy"]](
            query_vec, field, int(row["threshold"]), K)

        print("=" * 78)
        print(f"query: {row['query']} | strategy: {row['strategy']} | {row['mode']}")
        print(f"selectivity {row['selectivity'] * 100:.2f}% | "
              f"recall {row['recall_at_k']:.2f} | returned {row['n_returned']}/{K} | "
              f"GLS rho {row['gls_rho']:.3f}")

        print("\n  should have retrieved:")
        for doc_id, score in list(zip(truth_ids, truth_scores))[:3]:
            print(f"    [{score:.3f}] {meta.loc[doc_id, 'text'][:90]}")

        print("\n  actually retrieved:")
        for doc_id in got[:3]:
            score = float(embeddings[doc_id] @ query_vec)
            print(f"    [{score:.3f}] {meta.loc[doc_id, 'text'][:90]}")
        print()


# %% ================================================================= 10. MAIN

def main():
    print("=" * 78, "\n1. ENVIRONMENT\n", "=" * 78)
    record_environment()

    print("=" * 78, "\n2. DATASET\n", "=" * 78)
    meta = prepare_metadata(load_dataset())

    print("=" * 78, "\n3. EMBEDDINGS\n", "=" * 78)
    model, embeddings, embed_time = build_embeddings(meta)
    meta = add_correlated_bucket(meta, embeddings)

    print("=" * 78, "\n4. INDEX\n", "=" * 78)
    collection, build_time, size_mb, config = build_index(meta, embeddings)

    print("=" * 78, "\n5. QUERIES\n", "=" * 78)
    query_emb = model.encode(
        QUERIES, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    print("Queries:", len(QUERIES), "| shape:", query_emb.shape)

    print("=" * 78, "\n6. SELECTIVITY SWEEP\n", "=" * 78)
    strategies, over_fetch = make_strategies(collection, meta)
    results = run_sweep(meta, embeddings, query_emb, strategies)

    print("=" * 78, "\n7. ABLATION\n", "=" * 78)
    ablation = run_ablation(meta, embeddings, query_emb, over_fetch)

    print("=" * 78, "\n8. FIGURES\n", "=" * 78)
    make_figures(results, ablation)

    print("=" * 78, "\n9. ERROR ANALYSIS\n", "=" * 78)
    error_analysis(results, meta, embeddings, query_emb, strategies)

    print("=" * 78, "\n10. SUMMARY\n", "=" * 78)
    summary = (results.groupby(["mode", "strategy"])
               .agg(recall=("recall_at_k", "mean"),
                    completeness=("completeness_at_k", "mean"),
                    p50_ms=("lat_p50_ms", "mean"),
                    p95_ms=("lat_p95_ms", "mean"),
                    candidates=("candidates", "mean"))
               .round(3).reset_index())
    summary.to_csv(out("summary_table.csv"), index=False)
    print(summary.to_string(index=False))

    print(f"\nCorpus {len(meta)} docs | dim {embeddings.shape[1]} | K={K}")
    print(f"Embedding {embed_time:.1f}s | index build {build_time:.1f}s | "
          f"store {size_mb:.1f} MB")
    print("Index config:", config)
    print(f"\nAll outputs written to: {OUTPUT_DIR}")
    for name in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(out(name)) / 1024
        print(f"  {name:28s} {size:8.1f} KB")


if __name__ == "__main__":
    main()