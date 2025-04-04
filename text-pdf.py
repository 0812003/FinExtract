import PyPDF2
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A string containing the extracted text.
    """
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    """Preprocesses the extracted text.

    Args:
        text: The extracted text from the PDF.

    Returns:
        A list of preprocessed tokens.
    """
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Remove punctuation and special characters
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def extract_features(text):
    """Extracts features from the preprocessed text.

    Args:
        text: The preprocessed text.

    Returns:
        A list of features.
    """
    # Feature 1: Number of words
    num_words = len(text)

    # Feature 2: Number of unique words
    unique_words = set(text)
    num_unique_words = len(unique_words)

    # Feature 3: Average word length
    total_length = sum(len(word) for word in text)
    avg_word_length = total_length / num_words

    # Feature 4: Vocabulary richness (Type-Token Ratio)
    type_token_ratio = num_unique_words / num_words

    return [num_words, num_unique_words, avg_word_length, type_token_ratio]

def cluster_pdfs(pdf_paths):
    """Clusters PDF documents based on their extracted features.

    Args:
        pdf_paths: A list of paths to the PDF files.

    Returns:
        A list of cluster labels for each PDF.
    """
    features = []
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        # Handle missing text data (optional)
        if not text:
            text = ""  # Assign an empty string if no text is extracted

        preprocessed_text = preprocess_text(text)
        # Extract features and create a sub-list for each PDF
        extracted_features = extract_features(preprocessed_text)
        # Add the extracted text to the feature list
        extracted_features.append(text)
        features.append(extracted_features)

    # Ensure features list has the same length as the number of PDFs
    if len(features) != len(pdf_paths):
        print("Warning: Number of features does not match the number of PDFs.")

    # Create a DataFrame for the features
    df = pd.DataFrame(features, columns=['num_words', 'num_unique_words',
                                         'avg_word_length', 'type_token_ratio', 'text'])

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['text'])

    # Try different approaches to determine the optimal number of clusters
    # Option 1: Elbow method (example provided previously)
    # Option 2: Silhouette analysis (with handling for potential exceptions)
    silhouette_scores = []
    for k in range(2, min(10, len(pdf_paths))):  # Limit k to avoid exceeding the number of PDFs
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(tfidf_matrix)
            silhouette_avg = silhouette_score(tfidf_matrix, labels)
            silhouette_scores.append(silhouette_avg)
        except ValueError:  # Handle potential exception for insufficient labels
            silhouette_scores.append(-1)  # Assign a low score for cases with exceptions

    # Find the optimal number of clusters based on the highest silhouette score (or a heuristic)
    # You can also consider incorporating domain knowledge to choose the most relevant number of clusters
    if silhouette_scores:
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    else:
        # Handle cases where silhouette analysis fails (e.g., due to insufficient data)
        print("Warning: Silhouette analysis failed. Choosing k=2 as a default.")
        
    # Perform k-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(tfidf_matrix)

    return labels

def classify_structure_format(pdf_path):
    """Classifies the structure and format of a PDF document based on its content.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A dictionary containing the classification results.
    """
    text = extract_text_from_pdf(pdf_path)

    # Extract potential headers
    headers = re.findall(r'^\s*[A-Z]{2,}\s*$', text, flags=re.MULTILINE)

    # Identify column layouts based on regular expressions or visual analysis (using libraries like PyMuPDF or pdfminer)
    # (This part requires more advanced techniques and may involve manual inspection)

    # Identify naming characters (e.g., consistent use of underscores, hyphens, capitalization)
    naming_characters = re.findall(r'[_-]', pdf_path)

    # Create a dictionary to store the classification results
    classification = {
        'headers': headers,
        'column_layouts': 'Not yet implemented',
        'naming_characters': naming_characters
    }

    return classification

# Example usage:
pdf_paths = ['cashflow.pdf', 'balance sheet.pdf', 'income.pdf', 'comprehensive income.pdf', 'Shareholder Equity.pdf']
cluster_labels = cluster_pdfs(pdf_paths)

for pdf_path, label in zip(pdf_paths, cluster_labels):
    print(f"PDF: {pdf_path}, Cluster: {label}")
    print(classify_structure_format(pdf_path))

