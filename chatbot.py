import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load knowledge base
data = pd.read_csv("knowledge_base.csv")

questions = data["question"]
answers = data["answer"]

# Convert questions into vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

print("=== Retrieval Chatbot ===")
print("Type 'exit' to stop")

while True:
    user_query = input("\nYou: ")

    if user_query.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    query_vector = vectorizer.transform([user_query])

    similarity = cosine_similarity(query_vector, question_vectors)

    best_match = similarity.argmax()

    response = answers.iloc[best_match]

    print("Chatbot:", response)
