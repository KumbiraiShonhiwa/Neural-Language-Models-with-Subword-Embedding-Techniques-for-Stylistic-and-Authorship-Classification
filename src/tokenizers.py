import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
# Set path to your data directory
base_dir = '/content/drive/MyDrive/preprocessed/data'  # Update if needed
print(f"Looking for data in: {base_dir}")

# Function to read CSV files with Twitter data
def load_twitter_data(file_path):
    """Load Twitter data from CSV files."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        print("\nSample data:")
        print(df.head(3))

        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Process column names and structure
def process_twitter_dataframe(df):
    """Process and standardize Twitter dataframe structure."""

    # Check if the DataFrame is valid
    if df is None or len(df) == 0:
        print("Empty or invalid dataframe")
        return None

    # Case 1: If df has proper column names
    if 'text' in df.columns and 'user_id' in df.columns:
        print("DataFrame already has proper column names")
        return df

    # Case 2: If it has two columns but unnamed or numbered
    if df.shape[1] == 2:
        print("Renaming columns to 'text' and 'user_id'")
        df.columns = ['text', 'user_id']
        return df

    # This handles the case where columns are numeric (0,1) or unnamed
    print("Attempting to extract text and user_id from data...")

    # Identify which column contains the text
    text_col = None
    user_id_col = None

    # Look at first row to determine which column has text vs ID
    for col in df.columns:
        first_val = str(df[col].iloc[0])
        if first_val.startswith('b\'@') or first_val.startswith('b"@') or 'http' in first_val:
            text_col = col
        elif first_val.isdigit() or (first_val.isalnum() and len(first_val) > 7):
            user_id_col = col

    # If we found both columns, create a new DataFrame with proper column names
    if text_col is not None and user_id_col is not None:
        new_df = pd.DataFrame({
            'text': df[text_col],
            'user_id': df[user_id_col]
        })
        print(f"Created new DataFrame with 'text' and 'user_id' columns")
        return new_df

    # If columns not clearly identified, try first two columns
    if df.shape[1] >= 2:
        print("Using first two columns as text and user_id")
        new_df = pd.DataFrame({
            'text': df.iloc[:, 0],
            'user_id': df.iloc[:, 1]
        })
        return new_df

    print("Could not determine the proper structure of the dataframe")
    return None

# Clean and preprocess Twitter text
def clean_twitter_text(text):
    """Clean and normalize Twitter text for authorship attribution."""
    if pd.isna(text):
        return ""

    # Handle byte string format (b'text')
    if isinstance(text, str) and text.startswith("b'") or text.startswith('b"'):
        # Extract content from byte string representation
        text = text[2:-1]  # Remove b' and closing '

    # Convert to string if it's not already
    text = str(text)
    # Replace escaped newlines with space
    text = text.replace('\\n', ' ').replace('\n', ' ')
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    # Replace user mentions (keep as feature for authorship)
    text = re.sub(r'@\w+', '[USER]', text)
    # Replace hashtags (keep as feature for authorship)
    text = re.sub(r'#(\w+)', r'[HASHTAG] \1', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the data files
data_files = [
    os.path.join(base_dir, '100_users_with_200_tweets.csv'),
    os.path.join(base_dir, '200_users_with_200_tweets.csv'),
    os.path.join(base_dir, '500_users_with_200_tweets.csv')
]

# Load and combine data from all files
all_data = []
for file_path in data_files:
    if os.path.exists(file_path):
        df = load_twitter_data(file_path)
        if df is not None:
            processed_df = process_twitter_dataframe(df)
            if processed_df is not None:
                all_data.append(processed_df)
    else:
        print(f"File not found: {file_path}")

# Combine all dataframes
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined data: {combined_df.shape[0]} tweets from multiple files")
else:
    print("No valid data loaded. Please check the file paths and formats.")
    exit()

# Clean the text data
print("\nCleaning text data...")
combined_df['cleaned_text'] = combined_df['text'].apply(clean_twitter_text)

# Basic data analysis
print("\nBasic statistics:")
print(f"Total tweets: {len(combined_df)}")
print(f"Unique authors: {combined_df['user_id'].nunique()}")

# Calculate tweets per author
tweets_per_author = combined_df['user_id'].value_counts()
print(f"\nTweets per author statistics:")
print(f"Min: {tweets_per_author.min()}")
print(f"Max: {tweets_per_author.max()}")
print(f"Average: {tweets_per_author.mean():.2f}")

# Visualize tweets per author distribution
plt.figure(figsize=(10, 6))
sns.histplot(tweets_per_author, kde=True)
plt.title('Distribution of Tweets per Author')
plt.xlabel('Number of Tweets')
plt.ylabel('Number of Authors')
plt.savefig('tweets_per_author.png')
plt.close()

# Filter authors with a minimum number of tweets for reliable attribution
min_tweets = 50  # Authors should have at least 50 tweets for reliable attribution
authors_to_keep = tweets_per_author[tweets_per_author >= min_tweets].index.tolist()
filtered_df = combined_df[combined_df['user_id'].isin(authors_to_keep)].copy()

print(f"\nFiltered to {len(filtered_df)} tweets from {len(authors_to_keep)} authors with {min_tweets}+ tweets each")

# Determine which authors to use for the final dataset
# For computational feasibility, we'll limit to a maximum number of authors
max_authors = 100  # Adjust based on computational resources
if len(authors_to_keep) > max_authors:
    # Sort authors by number of tweets (descending) and take the top max_authors
    top_authors = tweets_per_author.nlargest(max_authors).index.tolist()
    filtered_df = combined_df[combined_df['user_id'].isin(top_authors)].copy()
    print(f"Limited to top {max_authors} authors with most tweets")

# Ensure balanced dataset by sampling the same number of tweets per author
tweets_per_author_after_filtering = filtered_df['user_id'].value_counts()
min_tweets_per_author = tweets_per_author_after_filtering.min()
print(f"Minimum tweets per author after filtering: {min_tweets_per_author}")

# Sample equal number of tweets per author
balanced_df = pd.DataFrame()
for author in filtered_df['user_id'].unique():
    author_tweets = filtered_df[filtered_df['user_id'] == author]
    # Sample min_tweets_per_author or all tweets if less
    if len(author_tweets) > min_tweets_per_author:
        sampled_tweets = author_tweets.sample(min_tweets_per_author, random_state=42)
    else:
        sampled_tweets = author_tweets
    balanced_df = pd.concat([balanced_df, sampled_tweets], ignore_index=True)

print(f"Balanced dataset: {len(balanced_df)} tweets from {balanced_df['user_id'].nunique()} authors")

# Split data into train, validation, and test sets
# Stratify by author to ensure all authors are represented in each split
train_df, temp_df = train_test_split(
    balanced_df,
    test_size=0.3,
    stratify=balanced_df['user_id'],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['user_id'],
    random_state=42
)

print(f"\nTrain set: {len(train_df)} tweets")
print(f"Validation set: {len(val_df)} tweets")
print(f"Test set: {len(test_df)} tweets")

# Verify author distribution across splits
print("\nVerifying author distribution across splits...")
train_authors = train_df['user_id'].value_counts()
val_authors = val_df['user_id'].value_counts()
test_authors = test_df['user_id'].value_counts()

print(f"Authors in train set: {len(train_authors)}")
print(f"Authors in validation set: {len(val_authors)}")
print(f"Authors in test set: {len(test_authors)}")

# Create author ID mapping (consistent across all splits)
all_authors = sorted(balanced_df['user_id'].unique())
author_to_id = {author: idx for idx, author in enumerate(all_authors)}

# Add numeric author IDs to dataframes
train_df['author_id'] = train_df['user_id'].map(author_to_id)
val_df['author_id'] = val_df['user_id'].map(author_to_id)
test_df['author_id'] = test_df['user_id'].map(author_to_id)

# Load tokenizers for different subword methods
print("\nLoading tokenizers...")
# 1. BPE (Byte-Pair Encoding)
xlm_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# 2. WordPiece
mbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# 3. SentencePiece
sentencepiece_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# Function to tokenize using different methods
def tokenize_text(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return tokens

# Sample data for tokenization comparison
print("\nTokenizing sample data with different methods...")
sample_size = min(100, len(train_df))
sample_data = train_df.sample(sample_size)

# Apply tokenization methods
sample_data['bpe_tokens'] = sample_data['cleaned_text'].apply(lambda x: tokenize_text(x, xlm_tokenizer))
sample_data['wordpiece_tokens'] = sample_data['cleaned_text'].apply(lambda x: tokenize_text(x, mbert_tokenizer))
sample_data['sentencepiece_tokens'] = sample_data['cleaned_text'].apply(lambda x: tokenize_text(x, sentencepiece_tokenizer))

# Calculate token counts
sample_data['bpe_token_count'] = sample_data['bpe_tokens'].apply(len)
sample_data['wordpiece_token_count'] = sample_data['wordpiece_tokens'].apply(len)
sample_data['sentencepiece_token_count'] = sample_data['sentencepiece_tokens'].apply(len)

# Print token count statistics
print("\nToken count statistics:")
for method in ['bpe', 'wordpiece', 'sentencepiece']:
    counts = sample_data[f'{method}_token_count']
    print(f"{method.capitalize()} tokens: min={counts.min()}, max={counts.max()}, avg={counts.mean():.2f}")

# Visualize token count distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(sample_data['bpe_token_count'], kde=True)
plt.title('BPE Token Count Distribution')

plt.subplot(1, 3, 2)
sns.histplot(sample_data['wordpiece_token_count'], kde=True)
plt.title('WordPiece Token Count Distribution')

plt.subplot(1, 3, 3)
sns.histplot(sample_data['sentencepiece_token_count'], kde=True)
plt.title('SentencePiece Token Count Distribution')

plt.tight_layout()
plt.savefig('token_distributions.png')
plt.close()

# Character n-gram analysis (important for authorship attribution)
def extract_ngrams(text, n=2):
    """Extract character n-grams from text"""
    text = str(text).lower()
    return [text[i:i+n] for i in range(len(text)-n+1)]

# Add n-gram extraction to a sample of the data
print("\nPerforming n-gram analysis for authorship features...")
ngram_sample = train_df.sample(min(500, len(train_df)))
ngram_sample['char_2grams'] = ngram_sample['cleaned_text'].apply(lambda x: extract_ngrams(x, 2))
ngram_sample['char_3grams'] = ngram_sample['cleaned_text'].apply(lambda x: extract_ngrams(x, 3))
ngram_sample['char_4grams'] = ngram_sample['cleaned_text'].apply(lambda x: extract_ngrams(x, 4))

# Count the most common n-grams
all_2grams = [gram for sublist in ngram_sample['char_2grams'] for gram in sublist]
all_3grams = [gram for sublist in ngram_sample['char_3grams'] for gram in sublist]
all_4grams = [gram for sublist in ngram_sample['char_4grams'] for gram in sublist]

# Get the 20 most common n-grams
common_2grams = Counter(all_2grams).most_common(20)
common_3grams = Counter(all_3grams).most_common(20)
common_4grams = Counter(all_4grams).most_common(20)

# Plot the most common n-grams
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.barplot(x=[x[0] for x in common_2grams], y=[x[1] for x in common_2grams])
plt.title('Most Common Character 2-grams')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
sns.barplot(x=[x[0] for x in common_3grams], y=[x[1] for x in common_3grams])
plt.title('Most Common Character 3-grams')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
sns.barplot(x=[x[0] for x in common_4grams], y=[x[1] for x in common_4grams])
plt.title('Most Common Character 4-grams')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('common_ngrams.png')
plt.close()

# Analyze authorship patterns - find distinctive n-grams for authors
print("\nAnalyzing distinctive n-grams by author...")

# Function to calculate author distinctiveness based on n-grams
def get_author_distinctive_ngrams(df, n=3, top_k=10):
    """Find the most distinctive n-grams for each author"""
    # Add n-gram column if it doesn't exist
    ngram_col = f'char_{n}grams'
    if ngram_col not in df.columns:
        df[ngram_col] = df['cleaned_text'].apply(lambda x: extract_ngrams(x, n))

    # Get global n-gram frequencies
    all_ngrams = [gram for sublist in df[ngram_col] for gram in sublist]
    global_freq = Counter(all_ngrams)
    total_ngrams = sum(global_freq.values())
    global_prob = {gram: count/total_ngrams for gram, count in global_freq.items()}

    # Find distinctive n-grams for each author
    author_distinctive_ngrams = {}

    for author in df['user_id'].unique():
        author_texts = df[df['user_id'] == author]
        author_ngrams = [gram for sublist in author_texts[ngram_col] for gram in sublist]
        author_freq = Counter(author_ngrams)
        author_total = sum(author_freq.values())

        # Calculate distinctiveness score (ratio of author frequency to global frequency)
        ngram_scores = {}
        for gram, count in author_freq.items():
            author_prob = count / author_total
            global_prob_gram = global_prob.get(gram, 1e-10)  # Avoid division by zero
            distinctiveness = author_prob / global_prob_gram
            ngram_scores[gram] = distinctiveness

        # Get top k distinctive n-grams
        top_ngrams = sorted(ngram_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        author_distinctive_ngrams[author] = top_ngrams

    return author_distinctive_ngrams

# Get distinctive 3-grams for a sample of authors
sample_authors = list(train_df['user_id'].unique())[:5]  # Just first 5 authors for demonstration
sample_author_df = train_df[train_df['user_id'].isin(sample_authors)]
distinctive_3grams = get_author_distinctive_ngrams(sample_author_df, n=3, top_k=10)

# Visualize distinctive n-grams for sample authors
plt.figure(figsize=(15, 10))
for i, (author, ngrams) in enumerate(distinctive_3grams.items()):
    if i >= 4:  # Only show first 4 authors to avoid crowding
        break
    plt.subplot(2, 2, i+1)
    plt.bar([x[0] for x in ngrams[:8]], [x[1] for x in ngrams[:8]])
    plt.title(f'Author {author}: Distinctive 3-grams')
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.savefig('author_distinctive_ngrams.png')
plt.close()

# Function to prepare data for transformer models
def prepare_data_for_transformers(df, tokenizer, max_length=128):
    """Convert dataframe to tokenized inputs for transformer models"""
    # Ensure all authors have numeric IDs
    if 'author_id' not in df.columns:
        all_authors = sorted(df['user_id'].unique())
        author_to_id = {author: idx for idx, author in enumerate(all_authors)}
        df['author_id'] = df['user_id'].map(author_to_id)

    # Get author mapping (even if already exists, for reference)
    author_map = {author: id for author, id in zip(df['user_id'], df['author_id'])}

    # Tokenize texts
    encodings = tokenizer(
        df['cleaned_text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    # Create labels tensor
    labels = torch.tensor(df['author_id'].values)

    return {
        'input_ids': encodings.input_ids,
        'attention_mask': encodings.attention_mask,
        'labels': labels,
        'author_map': author_map
    }

# Create datasets for each tokenization method
print("\nPreparing datasets for transformer models...")

print("Processing with BPE tokenization (XLM-RoBERTa)...")
bpe_train = prepare_data_for_transformers(train_df, xlm_tokenizer)
bpe_val = prepare_data_for_transformers(val_df, xlm_tokenizer)
bpe_test = prepare_data_for_transformers(test_df, xlm_tokenizer)

print("Processing with WordPiece tokenization (mBERT)...")
wordpiece_train = prepare_data_for_transformers(train_df, mbert_tokenizer)
wordpiece_val = prepare_data_for_transformers(val_df, mbert_tokenizer)
wordpiece_test = prepare_data_for_transformers(test_df, mbert_tokenizer)

print("Processing with SentencePiece tokenization (XLM-RoBERTa-Large)...")
sentencepiece_train = prepare_data_for_transformers(train_df, sentencepiece_tokenizer)
sentencepiece_val = prepare_data_for_transformers(val_df, sentencepiece_tokenizer)
sentencepiece_test = prepare_data_for_transformers(test_df, sentencepiece_tokenizer)

# Save the preprocessed datasets
print("\nSaving preprocessed datasets...")

# Create directory for output if it doesn't exist
output_dir = "/content/drive/MyDrive/preprocessed_output"
os.makedirs(output_dir, exist_ok=True)

# Save BPE datasets
torch.save(bpe_train, os.path.join(output_dir, 'bpe_train_dataset.pt'))
torch.save(bpe_val, os.path.join(output_dir, 'bpe_val_dataset.pt'))
torch.save(bpe_test, os.path.join(output_dir, 'bpe_test_dataset.pt'))

# Save WordPiece datasets
torch.save(wordpiece_train, os.path.join(output_dir, 'wordpiece_train_dataset.pt'))
torch.save(wordpiece_val, os.path.join(output_dir, 'wordpiece_val_dataset.pt'))
torch.save(wordpiece_test, os.path.join(output_dir, 'wordpiece_test_dataset.pt'))

# Save SentencePiece datasets
torch.save(sentencepiece_train, os.path.join(output_dir, 'sentencepiece_train_dataset.pt'))
torch.save(sentencepiece_val, os.path.join(output_dir, 'sentencepiece_val_dataset.pt'))
torch.save(sentencepiece_test, os.path.join(output_dir, 'sentencepiece_test_dataset.pt'))

# Save original processed DataFrames for reference
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

# Calculate author statistics for the README
author_tweet_counts = balanced_df['user_id'].value_counts()
min_count = author_tweet_counts.min()
max_count = author_tweet_counts.max()
avg_count = author_tweet_counts.mean()
num_authors = len(author_tweet_counts)


print(f"\nPreprocessing complete! All datasets saved to {output_dir}")
print("\nDatasets have been preprocessed using three different subword tokenization methods:")
print("1. Byte-Pair Encoding (BPE) - XLM-RoBERTa Base")
print("2. WordPiece - mBERT")
print("3. SentencePiece - XLM-RoBERTa Large")
print("\nEach dataset includes:")
print("- Tokenized text data")
print("- Real author ID mappings")
print("- Ready-to-use format for transformer models")