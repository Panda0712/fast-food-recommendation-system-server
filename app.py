import os
import pandas as pd
import spacy

from flask import Flask, request, jsonify
from flask_cors import CORS
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
CORS(app)  # Cho phép mọi nguồn gửi yêu cầu

nlp = spacy.load("en_core_web_sm")

url = 'https://iizqqbdczorzypgneais.supabase.co'
key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlpenFxYmRjem9yenlwZ25lYWlzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjYzNzM0MDMsImV4cCI6MjA0MTk0OTQwM30.9oRoFnOjJapMUTPI19Q6UP1_XQvRm-TyL7u57Q_2x6s'

supabase = create_client(url,key)

food_data = supabase.table('foods').select('*').execute()
train_food_data = pd.DataFrame(food_data.data)

def clean_and_extract_tags(text):
    doc = nlp(text.lower())
    tags = [token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
    return ', '.join(tags)

columns_to_extract_tags_from = ['category', 'name', 'description']

for column in columns_to_extract_tags_from:
    train_food_data[column] = train_food_data[column].apply(clean_and_extract_tags)

# Rating-based recommendation function
@app.route('/api/rating-based', methods=['POST'])
def rating_based_recommendation():
    try:
        # Calculate average ratings
        average_ratings = train_food_data.groupby(['id','name','regularPrice','discount', 'reviews_count', 'category', 'image'])['rating'].mean().reset_index()

        # Sort items by rating in descending order
        top_rated_items = average_ratings.sort_values(by='rating', ascending=False)

        # Select top 10 items
        top_items = top_rated_items.head(10).to_dict(orient='records')
        return jsonify(top_items)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Content-based recommendation function
@app.route('/api/content-based', methods=['POST'])
def content_based_recommendations():
    try:
        # Get item_name and top_n from request body
        data = request.json
        item_name = data.get('item_name', '')
        top_n = data.get('top_n', 10)

        # Check if the item name exists in the training data
        if item_name not in train_food_data['name'].values:
            return jsonify({'error': f"Item '{item_name}' not found in the training data."}), 404

        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_food_data['description'])
        cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

        # Find the index of the item
        item_index = train_food_data[train_food_data['name'] == item_name].index[0]

        # Get similar items
        similar_items = list(enumerate(cosine_similarities_content[item_index]))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        top_similar_items = similar_items[1:top_n + 1]
        recommended_item_indices = [x[0] for x in top_similar_items]

        # Return recommendations
        recommended_items = train_food_data.iloc[recommended_item_indices][['id','name', 'reviews_count', 'category', 'image', 'rating']].to_dict(orient='records')
        return jsonify(recommended_items)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Content-based recommendation with user data
@app.route('/api/user-content-based', methods=['POST'])
def user_content_based_recommendations():
    try:
        # Get guestId and top_n from request body
        data = request.json
        guest_id = data.get('guest_id', None)
        top_n = data.get('top_n', 10)

        if guest_id is None:
            return jsonify({'error': 'guest_id is required'}), 400

        # Get the list of foodIds ordered by the user
        user_orders = supabase.table('orders').select('*').eq('guestId', guest_id).execute()
        user_orders_data = pd.DataFrame(user_orders.data)

        if user_orders_data.empty:
            return jsonify({'error': f"No orders found for guest_id: {guest_id}"}), 404

        # Get the food details for the user's orders
        ordered_food_ids = user_orders_data['foodId'].unique()
        ordered_foods = train_food_data[train_food_data['id'].isin(ordered_food_ids)]

        if ordered_foods.empty:
            return jsonify({'error': 'No matching foods found for the user orders'}), 404

        # Create a user profile by combining descriptions of all ordered foods
        user_profile = ' '.join(ordered_foods['description'].values)

        # Vectorize all food descriptions and the user profile
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_food_data['description'])
        user_profile_vector = tfidf_vectorizer.transform([user_profile])

        # Calculate cosine similarity between user profile and all foods
        cosine_similarities = cosine_similarity(user_profile_vector, tfidf_matrix_content).flatten()

        # Get the indices of the top N recommendations (excluding already ordered foods)
        recommended_indices = [
            idx for idx, score in sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)
            if train_food_data.iloc[idx]['id'] not in ordered_food_ids
        ][:top_n]

        # Get the details of the recommended foods
        recommended_items = train_food_data.iloc[recommended_indices][['id','name','regularPrice','discount', 'category', 'description', 'rating', 'image']].to_dict(orient='records')

        return jsonify(recommended_items)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

