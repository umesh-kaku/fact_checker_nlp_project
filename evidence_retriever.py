from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading SciFact corpus...")

# Load corpus (documents)
corpus = load_dataset("scifact", "corpus")

documents = []
doc_ids = []

for row in corpus["train"]:
    doc_id = row["doc_id"]
    title = row["title"]
    abstract = " ".join(row["abstract"])

    text = title + " " + abstract

    documents.append(text)
    doc_ids.append(doc_id)

print("Building TF-IDF index...")

vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(documents)

print("Retriever ready")

# Retrieval function
def retrieve_evidence(claim, top_k=3):

    claim_vector = vectorizer.transform([claim])

    similarities = cosine_similarity(claim_vector, doc_vectors)[0]

    # Get top K documents
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for idx in top_indices:
        results.append({
            "doc_id": doc_ids[idx],
            "text": documents[idx],
            "score": similarities[idx]
        })

    return results