from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Sample data to search through (a list of dictionaries with titles and descriptions)
data = [
    {"title": "Python Programming", "description": "Learn Python programming from beginner to advanced."},
    {"title": "Flask Web Development", "description": "Create web applications with Flask."},
    {"title": "Data Science with Python", "description": "Analyze data and create models using Python."},
    {"title": "Machine Learning", "description": "Introduction to machine learning with Python."},
    {"title": "Web Development with JavaScript", "description": "Build dynamic websites using JavaScript."},
    {"title": "HTML & CSS Basics", "description": "Learn the fundamentals of web development."},
    {"title": "Introduction to SQL", "description": "Learn SQL for data querying and management."},
    {"title": "APIs and Microservices", "description": "Understand how to create and consume APIs."}
]

@app.route('/', methods=['GET', 'POST'])
def home():
    query = ""
    results = []
    scores = []
    
    if request.method == 'POST':
        query = request.form.get('search_query', '').lower()  # Get the search query
        documents = [f"{item['title']} {item['description']}" for item in data]  # Prepare the documents
        documents.append(query)  # Add the query to the document list
        
        # Create TF-IDF vectorizer and transform the documents
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        
        # Compute cosine similarity between the query and the documents
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Compare query with all documents
        scores = cosine_similarities.flatten()  # Flatten the result into a 1D array
        
        # Get the results sorted by score
        results = sorted(zip(data, scores), key=lambda x: x[1], reverse=True)

    return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)
