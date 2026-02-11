import pandas as pd
import numpy as np
import faiss
import argparse
import os
import re
import openai
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, kendalltau, spearmanr
from tqdm import tqdm

# ===================== DEFAULTS =====================

DEFAULT_TRAIN_QUERIES_PATH = "datasets/queries.train_filtered.tsv"
DEFAULT_TRAIN_SCORES_PATH = "dataset/train_query_mrr.tsv"
DEFAULT_EMBEDDINGS_DIR = "embeddings_output"
DEFAULT_DATASET_DIR = "dataset"

# Test datasets configuration: (test_queries_file, scores_file, score_column_name)
TEST_DATASETS = {
    "test2019": {
        "queries": "test2019-queries-filterd.tsv",
        "scores": "dev2019_query_NDCG.tsv",
    },
    "test2020": {
        "queries": "test2020-queries-filterd.tsv",
        "scores": "dev2020_query_NDCG.tsv",
    },
    "testhard": {
        "queries": "testhard-queries-filterd.tsv",
        "scores": "dlhard_ndcg10.tsv",
    },
    "testdev": {
        "queries": "testdev-queries-filterd.tsv",
        "scores": "dev_query_mrr.tsv",
    },
}

DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_NUM_FEW_SHOT_EXAMPLES = 5
DEFAULT_BATCH_SIZE = 1
DEFAULT_EMBEDDING_BATCH_SIZE = 256
DEFAULT_OLLAMA_API_KEY = 'ollama'
DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434/v1'

# ===================== PROMPTS =====================

SYSTEM_PROMPT = """You are a Query Performance Prediction (QPP) assistant. 
Your task is to predict the retrieval performance score (between 0 and 1) for a given query based on similar queries and their known performance scores.
The score represents how well a search system would perform for that query (higher is better).
\n/no_think"""

USER_PROMPT_TEMPLATE = """Based on the following similar queries and their performance scores, predict the performance score for the target query.

Similar queries and their scores:
{few_shot_examples}

Target query: "{target_query}"

Predict the performance score for the target query. Return ONLY a single float number between 0 and 1.
Do not explain, just return the number.
Score:"""

# ===================== HELPER FUNCTIONS =====================

def load_train_data(args):
    """Load training queries and scores."""
    print("Loading training queries...")
    train_queries_df = pd.read_csv(
        args.train_queries_path,
        sep='\t',
        names=['qid', 'query'],
        engine='pyarrow'
    )
    
    print("Loading training scores...")
    # Train scores has format: qid, queryText, score
    train_scores_df = pd.read_csv(
        args.train_scores_path,
        sep='\t',
        names=['qid', 'queryText', 'score'],
        engine='pyarrow'
    )
    
    # Merge queries with scores
    train_df = train_queries_df.merge(
        train_scores_df[['qid', 'score']], 
        on='qid', 
        how='inner'
    )
    
    print(f"Loaded {len(train_df)} training queries with scores")
    return train_df


def load_test_data(dataset_name, args):
    """Load test queries and their ground truth scores."""
    config = TEST_DATASETS[dataset_name]
    
    # Load queries
    queries_path = os.path.join(args.dataset_dir, config["queries"])
    queries_df = pd.read_csv(
        queries_path,
        sep='\t',
        names=['qid', 'query'],
        engine='pyarrow'
    )
    
    # Load scores
    scores_path = os.path.join(args.dataset_dir, config["scores"])
    scores_df = pd.read_csv(
        scores_path,
        sep='\t',
        names=['qid', 'score'],
        engine='pyarrow'
    )
    
    # Merge
    test_df = queries_df.merge(scores_df, on='qid', how='inner')
    
    print(f"Loaded {len(test_df)} test queries for {dataset_name}")
    return test_df


def build_faiss_index(args, embedding_model=None):
    """Build FAISS index from training queries."""
    print("Building FAISS index from training queries...")
    
    os.makedirs(args.embeddings_dir, exist_ok=True)
    
    # Load queries
    print(f"Loading queries from {args.train_queries_path}...")
    df = pd.read_csv(
        args.train_queries_path, 
        sep="\t", 
        header=None, 
        names=["qid", "query"], 
        engine='pyarrow'
    )
    query_ids = df["qid"].tolist()
    queries = df["query"].tolist()
    print(f"Loaded {len(queries)} queries")
    
    # Load model if not provided
    if embedding_model is None:
        print(f"Loading embedding model: {args.embedding_model}...")
        embedding_model = SentenceTransformer(args.embedding_model)
    
    # Encode queries
    print("Encoding queries...")
    embeddings = embedding_model.encode(
        queries,
        batch_size=args.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings = embeddings.astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors (cosine similarity)
    index.add(embeddings)
    print(f"FAISS index contains {index.ntotal} vectors")
    
    # Save embeddings and query IDs
    embeddings_path = os.path.join(args.embeddings_dir, "query_embeddings.npy")
    qids_path = os.path.join(args.embeddings_dir, "query_ids.npy")
    faiss_path = os.path.join(args.embeddings_dir, "query_index.faiss")
    
    print(f"Saving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings)
    
    print(f"Saving query IDs to {qids_path}...")
    np.save(qids_path, np.array(query_ids))
    
    print(f"Saving FAISS index to {faiss_path}...")
    faiss.write_index(index, faiss_path)
    
    print(f"Done! Output files saved to {args.embeddings_dir}/")
    
    return index, embeddings, np.array(query_ids)


def load_faiss_index(args, embedding_model=None):
    """Load FAISS index and related data. Build if files don't exist."""
    embeddings_path = os.path.join(args.embeddings_dir, "query_embeddings.npy")
    qids_path = os.path.join(args.embeddings_dir, "query_ids.npy")
    faiss_path = os.path.join(args.embeddings_dir, "query_index.faiss")
    
    # Check if all required files exist
    if not all(os.path.exists(p) for p in [embeddings_path, qids_path, faiss_path]):
        print("FAISS index files not found. Building index...")
        return build_faiss_index(args, embedding_model)
    
    print("Loading FAISS index and embeddings...")
    embeddings = np.load(embeddings_path)
    query_ids = np.load(qids_path)
    index = faiss.read_index(faiss_path)
    
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index, embeddings, query_ids


def get_similar_queries(query, model, index, train_df, faiss_query_ids, train_qid_to_row, k=DEFAULT_NUM_FEW_SHOT_EXAMPLES):
    """Find k similar queries from training set using FAISS."""
    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = query_embedding.astype(np.float32)
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, k + 10)  # Get more to filter
    
    similar_queries = []
    for dist, idx in zip(distances[0], indices[0]):
        if len(similar_queries) >= k:
            break
        if idx < len(faiss_query_ids):
            # Get the qid from FAISS index
            qid = faiss_query_ids[idx]
            # Look up in train_df using the qid
            if qid not in train_qid_to_row:
                continue
            row = train_qid_to_row[qid]
            # Skip if score is not valid
            if pd.isna(row['score']):
                continue
            similar_queries.append({
                'query': row['query'],
                'score': row['score'],
                'similarity': float(dist)
            })
    
    return similar_queries


def format_few_shot_examples(similar_queries):
    """Format similar queries for the prompt."""
    examples = []
    for i, sq in enumerate(similar_queries, 1):
        examples.append(f"{i}. Query: \"{sq['query']}\" -> Score: {sq['score']:.4f}")
    return "\n".join(examples)


def extract_score(response_text):
    """Extract float score from LLM response."""
    # Clean up response
    text = response_text.strip()
    
    # Handle thinking tags if present
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()
    
    # Remove any non-numeric characters except . and -
    text = text.replace('`', '').strip()
    
    # Try to find a float in the response
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    
    if matches:
        score = float(matches[0])
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    return None


def predict_score(client, model_name, target_query, similar_queries):
    """Use LLM to predict score for target query."""
    few_shot_text = format_few_shot_examples(similar_queries)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            few_shot_examples=few_shot_text,
            target_query=target_query
        )}
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                top_p=1
            )
            
            response_text = response.choices[0].message.content
            score = extract_score(response_text)
            
            if score is not None:
                return score, response_text
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
    
    return None, None


def calculate_correlations(predictions, ground_truth):
    """Calculate Pearson, Kendall, and Spearman correlations."""
    # Filter out None predictions
    valid_mask = [p is not None for p in predictions]
    preds = [p for p, v in zip(predictions, valid_mask) if v]
    gts = [g for g, v in zip(ground_truth, valid_mask) if v]
    
    if len(preds) < 2:
        return None, None, None
    
    preds = np.array(preds)
    gts = np.array(gts)
    
    pearson_corr, _ = pearsonr(preds, gts)
    kendall_corr, _ = kendalltau(preds, gts)
    spearman_corr, _ = spearmanr(preds, gts)
    
    return pearson_corr, kendall_corr, spearman_corr


def run_prediction(args):
    """Main function to run few-shot prediction."""
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize OpenAI client (Ollama)
    client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)
    
    # Load embedding model
    print(f"Loading embedding model: {args.embedding_model}...")
    embedding_model = SentenceTransformer(args.embedding_model)
    
    # Load FAISS index (will build if not exists, reusing the embedding model)
    index, embeddings, query_ids = load_faiss_index(args, embedding_model)
    
    # Load training data
    train_df = load_train_data(args)
    
    # Create mapping from qid to row data for train data
    train_qid_to_row = {row['qid']: row for _, row in train_df.iterrows()}
    
    # Process each test dataset
    test_datasets = args.datasets if args.datasets else list(TEST_DATASETS.keys())
    
    all_results = {}
    
    for dataset_name in test_datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load test data
        test_df = load_test_data(dataset_name, args)
        
        predictions = []
        ground_truths = []
        results = []
        
        # Check for existing results to resume
        output_file = os.path.join(args.output_dir, f"predictions_{dataset_name}_{args.model.replace(':', '_').replace('/', '_')}.tsv")
        seen_qids = set()
        
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file, sep='\t')
            seen_qids = set(existing_df['qid'].values)
            results = existing_df.to_dict('records')
            predictions = existing_df['prediction'].tolist()
            ground_truths = existing_df['ground_truth'].tolist()
            print(f"Resuming from {len(seen_qids)} existing predictions")
        
        # Process each query
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Predicting {dataset_name}"):
            qid = row['qid']
            query = row['query']
            ground_truth = row['score']
            
            if qid in seen_qids:
                continue
            
            # Find similar queries
            similar_queries = get_similar_queries(
                query, embedding_model, index, train_df, query_ids, train_qid_to_row,
                k=args.num_few_shot
            )
            
            if len(similar_queries) == 0:
                print(f"Warning: No similar queries found for qid={qid}")
                predictions.append(None)
                ground_truths.append(ground_truth)
                results.append({
                    'qid': qid,
                    'query': query,
                    'ground_truth': ground_truth,
                    'prediction': None,
                    'similar_queries': ''
                })
                continue
            
            # Predict score
            predicted_score, raw_response = predict_score(
                client, args.model, query, similar_queries
            )
            
            predictions.append(predicted_score)
            ground_truths.append(ground_truth)
            
            # Store result
            similar_queries_str = '|'.join([f"{sq['query']}:{sq['score']:.4f}" for sq in similar_queries])
            results.append({
                'qid': qid,
                'query': query,
                'ground_truth': ground_truth,
                'prediction': predicted_score,
                'similar_queries': similar_queries_str
            })
            
            # Save incrementally
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, sep='\t', index=False)
        
        # Calculate correlations
        pearson, kendall, spearman = calculate_correlations(predictions, ground_truths)
        
        print(f"\n--- Results for {dataset_name} ---")
        print(f"Total queries: {len(test_df)}")
        print(f"Valid predictions: {sum(1 for p in predictions if p is not None)}")
        print(f"Pearson correlation: {pearson:.4f}" if pearson else "Pearson correlation: N/A")
        print(f"Kendall correlation: {kendall:.4f}" if kendall else "Kendall correlation: N/A")
        print(f"Spearman correlation: {spearman:.4f}" if spearman else "Spearman correlation: N/A")
        
        all_results[dataset_name] = {
            'pearson': pearson,
            'kendall': kendall,
            'spearman': spearman,
            'total_queries': len(test_df),
            'valid_predictions': sum(1 for p in predictions if p is not None)
        }
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f"summary_{args.model.replace(':', '_').replace('/', '_')}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Few-Shot QPP Results\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Number of few-shot examples: {args.num_few_shot}\n")
        f.write(f"{'='*60}\n\n")
        
        for dataset_name, metrics in all_results.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"  Total queries: {metrics['total_queries']}\n")
            f.write(f"  Valid predictions: {metrics['valid_predictions']}\n")
            f.write(f"  Pearson: {metrics['pearson']:.4f}\n" if metrics['pearson'] else "  Pearson: N/A\n")
            f.write(f"  Kendall: {metrics['kendall']:.4f}\n" if metrics['kendall'] else "  Kendall: N/A\n")
            f.write(f"  Spearman: {metrics['spearman']:.4f}\n" if metrics['spearman'] else "  Spearman: N/A\n")
            f.write("\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Few-shot Query Performance Prediction with LLMs")
    
    # Required arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="LLM model name (e.g., phi4:latest, qwen3:8b)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory for predictions and results"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--datasets", "-d",
        type=str,
        nargs='+',
        default=None,
        choices=list(TEST_DATASETS.keys()),
        help="Test datasets to evaluate (default: all)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing test dataset files (default: {DEFAULT_DATASET_DIR})"
    )
    
    # Path arguments
    parser.add_argument(
        "--train_queries_path",
        type=str,
        default=DEFAULT_TRAIN_QUERIES_PATH,
        help=f"Path to training queries TSV (default: {DEFAULT_TRAIN_QUERIES_PATH})"
    )
    parser.add_argument(
        "--train_scores_path",
        type=str,
        default=DEFAULT_TRAIN_SCORES_PATH,
        help=f"Path to training scores TSV (default: {DEFAULT_TRAIN_SCORES_PATH})"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Directory for FAISS index and embeddings (default: {DEFAULT_EMBEDDINGS_DIR})"
    )
    
    # Model and few-shot settings
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help=f"Sentence-transformer embedding model name (default: {DEFAULT_EMBEDDING_MODEL_NAME})"
    )
    parser.add_argument(
        "--num_few_shot", "-k",
        type=int,
        default=DEFAULT_NUM_FEW_SHOT_EXAMPLES,
        help=f"Number of few-shot examples (default: {DEFAULT_NUM_FEW_SHOT_EXAMPLES})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for LLM inference (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help=f"Batch size for creating embeddings (default: {DEFAULT_EMBEDDING_BATCH_SIZE})"
    )
    
    # Ollama / API connection settings
    parser.add_argument(
        "--api_key",
        type=str,
        default=DEFAULT_OLLAMA_API_KEY,
        help=f"API key for the LLM server (default: {DEFAULT_OLLAMA_API_KEY})"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Base URL for the LLM server (default: {DEFAULT_OLLAMA_BASE_URL})"
    )
    
    args = parser.parse_args()
    
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {args.datasets if args.datasets else 'all'}")
    print(f"Few-shot examples: {args.num_few_shot}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Base URL: {args.base_url}")
    
    run_prediction(args)


if __name__ == "__main__":
    main()
