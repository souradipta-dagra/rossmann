import pandas as pd
import numpy as np
import regex as re
import unicodedata
from sentence_transformers import SentenceTransformer
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define valid languages and their codes
VALID_LANGUAGES = {
    'english': 'en', 'french': 'fr', 'italian': 'it',
    'kazakh': 'kk', 'rus': 'ru', 'spanish': 'es', 'swedish': 'sv'
}

# Required and metadata columns
REQUIRED_COLUMNS = ["project", "observation", "solution", "language"]
METADATA_COLUMNS = ['fleet', 'subsystem', 'database', 'observationcategory', 
                    'solutioncategory', 'failureclass', 'date']

def clean_text(text):
    """Clean and normalize text content"""
    if pd.isna(text):
        return ""
    text = unicodedata.normalize('NFKC', str(text).lower())
    text = re.sub(r'\s+', ' ', text)
    return re.sub(r'[^\p{L}\p{N}\s\-.,;:!?()/]', '', text, flags=re.UNICODE).strip()

def preprocess_text(df, text_columns):
    """Apply text cleaning to specified columns"""
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    return df

def standardize_language(lang):
    """Convert language names to standard codes"""
    return VALID_LANGUAGES.get(str(lang).strip().lower(), 'unknown') if pd.notna(lang) else 'unknown'

def fill_missing_values(df, project_id):
    """Fill missing values with mode for the project"""
    # Get mode values for key columns
    modes = {}
    for col in df.columns:
        if col in ["project", "database", "language"] and col in df.columns:
            if not df[col].empty:
                modes[col] = df[col].mode().iloc[0]
            else:
                modes[col] = "unknown"
    
    # If project is missing, use the project_id
    if "project" not in modes:
        modes["project"] = project_id
        
    # Fill missing values
    return df.fillna(modes)

def process_dataset(df, project_id=None):
    """Process a single dataset"""
    # Drop rows with missing required fields
    df = df.dropna(subset=["observation", "solution"])
    
    # If project_id is provided but missing in dataframe, add it
    if project_id and "project" not in df.columns:
        df["project"] = project_id
    elif "project" not in df.columns:
        df["project"] = "unknown"
    
    # Clean text columns
    text_columns = ["observation", "solution", "observationcategory", "solutioncategory"]
    df = preprocess_text(df, text_columns)
    
    # Standardize language
    if "language" in df.columns:
        df["language"] = df["language"].apply(standardize_language)
    else:
        df["language"] = "unknown"
    
    # Fill missing values
    df = fill_missing_values(df, project_id)
    
    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS + METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = "unknown"
    
    return df

def combine_datasets(dataframes):
    """Combine multiple preprocessed dataframes"""
    # Find common columns to keep data consistent
    common_columns = set.intersection(*[set(df.columns) for df in dataframes.values()]) | set(REQUIRED_COLUMNS)
    
    # Filter to common columns
    filtered_dfs = {name: df[list(common_columns)] for name, df in dataframes.items()}
    
    # Combine datasets
    combined_df = pd.concat(
        [df.fillna("").astype(str) for df in filtered_dfs.values()],
        ignore_index=True
    )
    
    # Create text column for searching
    combined_df['text'] = combined_df.apply(
        lambda x: f"Observation: {x['observation']} [SEP] Solution: {x['solution']}", axis=1
    ).str.strip()
    
    # Remove empty entries and duplicates
    combined_df = combined_df[combined_df['text'] != 'Observation:  [SEP] Solution: '].reset_index(drop=True)
    combined_df.drop_duplicates(subset=['text', 'project'], inplace=True)
    
    return combined_df

def generate_embeddings(df, model_name="multilingual-e5-large"):
    """Generate embeddings for text fields"""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info("Generating observation embeddings")
    observation_embeddings = model.encode(df["observation"].tolist(), show_progress_bar=True)
    
    logger.info("Generating solution embeddings")
    solution_embeddings = model.encode(df["solution"].tolist(), show_progress_bar=True)
    
    logger.info("Generating combined text embeddings")
    text_embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    
    return {
        "observation_embeddings": observation_embeddings,
        "solution_embeddings": solution_embeddings,
        "text_embeddings": text_embeddings
    }

def setup_qdrant(embeddings_dim=768):
    """Set up Qdrant collection"""
    client = QdrantClient("localhost", port=6333)
    
    # Create collection if it doesn't exist
    try:
        client.get_collection("alstom_kb")
        logger.info("Collection already exists")
    except:
        logger.info("Creating new collection")
        client.create_collection(
            collection_name="alstom_kb",
            vectors_config={
                "text": VectorParams(size=embeddings_dim, distance=Distance.COSINE),
                "observation": VectorParams(size=embeddings_dim, distance=Distance.COSINE),
                "solution": VectorParams(size=embeddings_dim, distance=Distance.COSINE)
            }
        )
    
    return client

def store_in_qdrant(df, embeddings, client):
    """Store embeddings and metadata in Qdrant"""
    logger.info(f"Storing {len(df)} entries in Qdrant")
    
    # Prepare points with vectors and payloads
    points = []
    for i, row in df.iterrows():
        points.append({
            "id": i,
            "vectors": {
                "text": embeddings["text_embeddings"][i].tolist(),
                "observation": embeddings["observation_embeddings"][i].tolist(),
                "solution": embeddings["solution_embeddings"][i].tolist()
            },
            "payload": {
                "observation": row["observation"],
                "solution": row["solution"],
                "project": row["project"],
                "language": row["language"],
                "database": row["database"] if "database" in row else "unknown",
                "fleet": row["fleet"] if "fleet" in row else "unknown",
                "subsystem": row["subsystem"] if "subsystem" in row else "unknown",
                "id": f"{row['project']}_{i}"
            }
        })
    
    # Store in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name="alstom_kb", points=batch)
        logger.info(f"Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
    
    return len(points)

def process_and_index_datasets(datasets_dict):
    """Process all datasets and index in Qdrant"""
    # Process each dataset
    processed_dfs = {}
    for name, df in datasets_dict.items():
        logger.info(f"Processing dataset: {name}")
        processed_dfs[name] = process_dataset(df, name)
    
    # Combine datasets
    logger.info("Combining datasets")
    combined_df = combine_datasets(processed_dfs)
    logger.info(f"Combined dataset size: {len(combined_df)} entries")
    
    # Generate embeddings
    embeddings = generate_embeddings(combined_df)
    
    # Set up Qdrant
    client = setup_qdrant(embeddings["text_embeddings"].shape[1])
    
    # Store in Qdrant
    stored_count = store_in_qdrant(combined_df, embeddings, client)
    logger.info(f"Successfully stored {stored_count} entries in Qdrant")
    
    return combined_df

def add_new_entries(new_df, project_id):
    """Process and add new entries to the system"""
    # Process the new dataset
    processed_df = process_dataset(new_df, project_id)
    
    # Generate embeddings
    embeddings = generate_embeddings(processed_df)
    
    # Set up Qdrant client
    client = QdrantClient("localhost", port=6333)
    
    # Store in Qdrant
    stored_count = store_in_qdrant(processed_df, embeddings, client)
    logger.info(f"Added {stored_count} new entries for project {project_id}")
    
    return stored_count

# Example usage
if __name__ == "__main__":
    # Load datasets (example)
    # Replace with your actual data loading logic
    dataset_names = ["sts_dubai", "sts_italy", "sts_kz8a"]
    datasets = {name: pd.read_csv(f"{name}.csv") for name in dataset_names}
    
    # Process and index
    combined_data = process_and_index_datasets(datasets)
    
    # Save combined data
    combined_data.to_parquet("alstom_combined_kb.parquet")
    logger.info("Preprocessing and indexing completed successfully")