
import re
from string import punctuation, digits
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import itertools
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from pathlib import Path

# Constants
NIFTY_WORDS = ['nifty', 'nifty50', 'dalalstreet', 'nse', 'bse', 'stock', 'market',
               'stocks', 'markets', 'share', 'shares', 'india', 'indian', "'"]

# Global variables
text = None
corp = None
filtered_text = None
sentiments = None
sentiment_df = None

# Load data
app_dir = Path(__file__).parent
data_file = app_dir / "Datasets/twitter_dummy_data.txt"
text = [line.rstrip() for line in open(data_file)]


# ============================================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_text_for_wordcloud():
    """Preprocess text for word cloud generation (aggressive cleaning)."""
    global corp, filtered_text

    # Convert to lowercase
    corp = [item.lower() for item in text]

    # Remove mentions, tags and URLs
    corp = [re.sub(r'@\w+', '', item) for item in corp]
    corp = [re.sub(r'#\w+', '', item) for item in corp]
    corp = [re.sub(r'http\S+', '', item) for item in corp]

    # Remove punctuation
    remove_punc = str.maketrans('', '', punctuation)
    corp = [item.translate(remove_punc) for item in corp]

    # Remove digits
    remove_digits = str.maketrans('', '', digits)
    corp = [item.translate(remove_digits) for item in corp]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    corp = [' '.join([word for word in item.split() if word not in stop_words]) for item in corp]

    # Remove nifty-related words
    corp = [' '.join([word for word in item.split() if word not in NIFTY_WORDS]) for item in corp]

    # Tokenize text
    fs = []
    for line in corp:
        words = word_tokenize(line)
        fs.append(words)

    filtered_text = list(itertools.chain.from_iterable(fs))

    return filtered_text


def preprocess_text_for_sentiment():
    """Preprocess text for sentiment analysis (lighter cleaning)."""
    # Convert to lowercase
    corp_sentiment = [line.lower() for line in text]

    # Remove mentions, dates and URLs
    corp_sentiment = [re.sub(r'@\w+', '', item) for item in corp_sentiment]
    corp_sentiment = [re.sub(r'http\S+', '', item) for item in corp_sentiment]
    corp_sentiment = [re.sub(r'^\[\d{4}-\d{2}-\d{2}\s+\d{4}\]\s*', '', item) for item in corp_sentiment]

    return corp_sentiment


# ============================================================================
# WORD CLOUD FUNCTIONS
# ============================================================================

def generate_wordcloud():
    """Generate and display word cloud from preprocessed text."""
    if filtered_text is None:
        preprocess_text_for_wordcloud()

    wordcloud = WordCloud(
        background_color="white",
        min_word_length=5,
        max_words=50,
        width=800,
        height=800
    ).generate(' '.join(filtered_text))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wordcloud)
    ax.axis("off")
    plt.tight_layout(pad=0)

    return fig


def get_top_words(n=20):
    """Get the top N most frequent words from the preprocessed text."""
    if filtered_text is None:
        preprocess_text_for_wordcloud()

    # Count word frequencies
    word_freq = {}
    for word in filtered_text:
        if len(word) >= 5:  # Only words with 5+ characters
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and get top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n]

    # Create DataFrame
    top_words_df = pd.DataFrame(sorted_words, columns=['Word', 'Frequency'])

    return top_words_df


# ============================================================================
# SENTIMENT ANALYSIS FUNCTIONS
# ============================================================================

def perform_sentiment_analysis():
    """Perform sentiment analysis using VADER."""
    global sentiments, sentiment_df

    sia = SentimentIntensityAnalyzer()
    corp_sentiment = preprocess_text_for_sentiment()

    sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

    # Analyze sentiment for each line
    for line in corp_sentiment:
        score = sia.polarity_scores(line)
        if score['compound'] >= 0.05:
            sentiments['Positive'] += 1
        elif score['compound'] <= -0.05:
            sentiments['Negative'] += 1
        else:
            sentiments['Neutral'] += 1

    # Create DataFrame
    sentiment_df = pd.DataFrame(list(sentiments.items()), columns=['Sentiment', 'Count'])

    # Add percentage
    total = sum(sentiments.values())
    sentiment_df['Percentage'] = (sentiment_df['Count'] / total * 100).round(1)

    return sentiment_df


def get_sentiment_summary():
    """Get sentiment analysis summary as a DataFrame."""
    if sentiment_df is None:
        perform_sentiment_analysis()

    return sentiment_df


def plot_sentiment_pie():
    """Visualize sentiment distribution with pie chart."""
    if sentiment_df is None:
        perform_sentiment_analysis()

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['green', 'red', 'gray']
    explode = (0.05, 0.05, 0.05)

    ax.pie(sentiment_df['Count'],
           labels=sentiment_df['Sentiment'],
           autopct='%1.1f%%',
           colors=colors,
           explode=explode,
           startangle=90,
           textprops={'fontsize': 12})

    ax.set_title('Sentiment Distribution of Nifty50 Tweets', fontsize=14, pad=20)

    plt.tight_layout()

    return fig


# ============================================================================
# SAMPLE TEXT FUNCTIONS
# ============================================================================

def get_sample_tweets(n=10):
    """Get sample tweets from the dataset."""
    sample_df = pd.DataFrame({
        'Tweet': text[:n]
    })
    sample_df.index = range(1, n + 1)
    sample_df.index.name = 'Tweet #'

    return sample_df


def get_preprocessing_example():
    """Show an example of text preprocessing steps."""
    if len(text) < 3:
        return None

    example_idx = 2
    original = text[example_idx]

    steps = []

    # Step 1: Lowercase
    step1 = original.lower()
    steps.append(('Original', original))
    steps.append(('Lowercase', step1))

    # Step 2: Remove mentions, tags, URLs
    step2 = re.sub(r'@\w+', '', step1)
    step2 = re.sub(r'#\w+', '', step2)
    step2 = re.sub(r'http\S+', '', step2)
    steps.append(('Remove mentions/tags/URLs', step2))

    # Step 3: Remove punctuation
    remove_punc = str.maketrans('', '', punctuation)
    step3 = step2.translate(remove_punc)
    steps.append(('Remove punctuation', step3))

    # Step 4: Remove digits
    remove_digits_trans = str.maketrans('', '', digits)
    step4 = step3.translate(remove_digits_trans)
    steps.append(('Remove digits', step4))

    # Step 5: Remove stopwords
    stop_words = set(stopwords.words('english'))
    step5 = ' '.join([word for word in step4.split() if word not in stop_words])
    steps.append(('Remove stopwords', step5))

    # Step 6: Remove nifty-related words
    step6 = ' '.join([word for word in step5.split() if word not in NIFTY_WORDS])
    steps.append(('Remove domain words', step6))

    example_df = pd.DataFrame(steps, columns=['Step', 'Text'])

    return example_df


# ============================================================================
# CODE SNIPPET FUNCTIONS
# ============================================================================

def text_preprocessing_code():
    return """
# Text Preprocessing for Word Cloud
import re
from string import punctuation, digits
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Convert to lowercase
corp = [item.lower() for item in text]

# Remove mentions, tags and URLs
corp = [re.sub(r'@\\w+', '', item) for item in corp]
corp = [re.sub(r'#\\w+', '', item) for item in corp]
corp = [re.sub(r'http\\S+', '', item) for item in corp]

# Remove punctuation
remove_punc = str.maketrans('', '', punctuation)
corp = [item.translate(remove_punc) for item in corp]

# Remove digits
remove_digits = str.maketrans('', '', digits)
corp = [item.translate(remove_digits) for item in corp]

# Remove stopwords
stop_words = set(stopwords.words('english'))
corp = [' '.join([word for word in item.split() if word not in stop_words]) for item in corp]

# Remove domain-specific words
nifty_words = ['nifty', 'nifty50', 'dalalstreet', 'nse', 'bse', 'stock', 'market']
corp = [' '.join([word for word in item.split() if word not in nifty_words]) for item in corp]

# Tokenize
fs = []
for line in corp:
    words = word_tokenize(line)
    fs.append(words)

filtered_text = list(itertools.chain.from_iterable(fs))
"""


def wordcloud_generation_code():
    return """
# Generate Word Cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(
    background_color="white",
    min_word_length=5,
    max_words=50
).generate(' '.join(filtered_text))

plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
"""


def sentiment_analysis_code():
    return """
# Sentiment Analysis using VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

sia = SentimentIntensityAnalyzer()

# Lighter preprocessing for sentiment
corp_sentiment = [line.lower() for line in text]
corp_sentiment = [re.sub(r'@\\w+', '', item) for item in corp_sentiment]
corp_sentiment = [re.sub(r'http\\S+', '', item) for item in corp_sentiment]

sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

# Analyze sentiment
for line in corp_sentiment:
    score = sia.polarity_scores(line)
    if score['compound'] >= 0.05:
        sentiments['Positive'] += 1
    elif score['compound'] <= -0.05:
        sentiments['Negative'] += 1
    else:
        sentiments['Neutral'] += 1

print(sentiments)
"""


def sentiment_visualization_code():
    return """
# Visualize Sentiment Distribution
import pandas as pd
import matplotlib.pyplot as plt

sentiment_df = pd.DataFrame(list(sentiments.items()), columns=['Sentiment', 'Count'])

plt.figure(figsize=(10, 6))
plt.bar(sentiment_df['Sentiment'], sentiment_df['Count'], color=['green', 'red', 'gray'])
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Sentiment Distribution of Nifty50 Tweets', fontsize=14)
plt.grid(axis='y', alpha=0.3)

# Add count labels
for i, v in enumerate(sentiment_df['Count']):
    plt.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
"""
