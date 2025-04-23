from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import uuid
import os
import ast
import warnings
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

# Configuration
UPLOAD_FOLDER = 'user_data'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the books dataset
def load_data():
    return pd.read_csv('BooksDataset.csv')

books_df = load_data()

# First Layer: User Segmentation Layer
class UserSegmentLayer:
    def __init__(self):
        self.questions = [
            "How often do you read books? (1-5)",
            "Rate your interest in fiction books (1-5)",
            "Rate your interest in non-fiction books (1-5)",
            "Rate your interest in mystery books (1-5)",
            "Rate your interest in science fiction books (1-5)",
            "Rate your interest in fantasy books (1-5)",
            "How important is the book's length to you? (1-5)",
            "How much do you care about book ratings? (1-5)",
            "Do you prefer books in English? (1-5)",
            "How important is the author's reputation to you? (1-5)",
            "Do you prefer recent books or classics? (1 for classics, 5 for recent)",
            "How much do you care about the book's publication year? (1-5)",
            "Rate your interest in foreign language books (1-5)",
            "How important are book reviews to you? (1-5)",
            "Do you prefer reading physical books or e-books? (1 for e-books, 5 for physical books)"
        ]
        self.scaler = StandardScaler()

    def segment_user(self, answers):
        avg_score = sum(answers) / len(answers)
        if avg_score < 2:
            return 0
        elif avg_score < 3:
            return 1
        elif avg_score < 4:
            return 2
        else:
            return 3

    def generate_user_id(self):
        return str(uuid.uuid4())

    def store_user_data(self, user_id, answers, segment):
        user_data = {
            'user_id': user_id,
            'answers': answers,
            'segment': int(segment)
        }
        os.makedirs('user_data', exist_ok=True)
        with open(f'user_data/{user_id}.pkl', 'wb') as f:
            pickle.dump(user_data, f)

# Second Layer: Recommendation Layer
class RecommendationLayer:
    def __init__(self, books_df):
        self.books_df = books_df
        self.prepare_data()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.train_model()

    def prepare_data(self):
        # Parse genres from the 'Category' column
        def parse_genres(category_str):
            try:
                genres = ast.literal_eval(category_str)
                if isinstance(genres, list) and all(isinstance(g, str) for g in genres):
                    return genres
                elif isinstance(genres, str):
                    return [genres]
                else:
                    return []
            except:
                return []

        self.books_df['genres'] = self.books_df['Category'].fillna('[]').apply(parse_genres)

        self.genre_list = ['Fiction', 'Non-Fiction', 'Mystery', 'Science Fiction', 'Fantasy']
        for genre in self.genre_list:
            self.books_df[f'genre_{genre}'] = self.books_df['genres'].apply(lambda x: 1 if genre in x else 0)

        # Convert publication year to numeric
        self.books_df['publication_year'] = pd.to_numeric(self.books_df['Publish Date (Year)'], errors='coerce')
        self.books_df['publication_year'] = self.books_df['publication_year'].fillna(
            self.books_df['publication_year'].median()
        )

        # Normalize numerical features
        numerical_features = ['Price Starting With ($)', 'publication_year']
        for feature in numerical_features:
            self.books_df[feature] = (self.books_df[feature] - self.books_df[feature].min()) / (
                        self.books_df[feature].max() - self.books_df[feature].min())

        self.features = [col for col in self.books_df.columns if col.startswith('genre_')] + ['Price Starting With ($)',
                                                                                           'publication_year']

        self.X = self.books_df[self.features].fillna(0)
        self.y = self.books_df['Price Starting With ($)']

    def train_model(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def get_user_segment(self, user_id):
        try:
            with open(f'user_data/{user_id}.pkl', 'rb') as f:
                user_data = pickle.load(f)
            return user_data['segment'], user_data['answers']
        except FileNotFoundError:
            return None, [3] * 15  # Default average preferences if user not found

    def recommend_books(self, user_id, top_n=10):
        _, user_answers = self.get_user_segment(user_id)

        user_profile = {
            'read_frequency': user_answers[0],
            'genre_preferences': user_answers[1:6],
            'length_importance': user_answers[6],
            'rating_importance': user_answers[7],
            'english_preference': user_answers[8],
            'author_importance': user_answers[9],
            'recency_preference': user_answers[10],
            'publication_year_importance': user_answers[11],
            'foreign_preference': user_answers[12],
            'reviews_importance': user_answers[13],
            'format_preference': user_answers[14]
        }

        # Calculate scores with configurable weights
        genre_weight = 0.4
        price_weight = 0.3
        year_weight = 0.3

        self.books_df['user_score'] = 0

        # Genre preferences
        for i, genre in enumerate(self.genre_list):
            self.books_df['user_score'] += (
                self.books_df[f'genre_{genre}'] *
                user_profile['genre_preferences'][i] *
                genre_weight
            )

        # Other factors
        self.books_df['user_score'] += (
            self.books_df['Price Starting With ($)'] *
            user_profile['length_importance'] *
            price_weight
        )
        self.books_df['user_score'] += (
            self.books_df['publication_year'] *
            user_profile['recency_preference'] *
            year_weight
        )

        # Normalize scores
        self.books_df['user_score'] = (
            (self.books_df['user_score'] - self.books_df['user_score'].min()) /
            (self.books_df['user_score'].max() - self.books_df['user_score'].min())
        )

        self.books_df['final_score'] = (
            0.7 * self.books_df['user_score'] +
            0.3 * self.books_df['Price Starting With ($)']
        )

        recommended_books = self.books_df.sort_values('final_score', ascending=False).head(top_n)
        return recommended_books[['Title', 'final_score', 'genres', 'Price Starting With ($)', 'publication_year']]

# Initialize layers
user_segment_layer = UserSegmentLayer()
recommendation_layer = RecommendationLayer(books_df)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        answers = [int(request.form[f'q{i}']) for i in range(len(user_segment_layer.questions))]
        segment = user_segment_layer.segment_user(answers)
        user_id = user_segment_layer.generate_user_id()
        user_segment_layer.store_user_data(user_id, answers, segment)
        session['user_id'] = user_id
        return redirect(url_for('recommendations'))
    return render_template('survey.html', questions=user_segment_layer.questions)

@app.route('/recommendations')
def recommendations():
    if 'user_id' not in session:
        flash('Please complete the survey first', 'warning')
        return redirect(url_for('survey'))
    
    user_id = session['user_id']
    recommendations = recommendation_layer.recommend_books(user_id)
    
    # Convert recommendations to a list of dicts for the template
    recommended_books = []
    for _, row in recommendations.iterrows():
        recommended_books.append({
            'title': row['Title'],
            'score': f"{row['final_score']:.2f}",
            'genres': ', '.join(row['genres']),
            'price': f"${row['Price Starting With ($)']:.2f}",
            'year': int(row['publication_year'])
        })
    
    return render_template('recommendations.html', books=recommended_books)

@app.route('/new_user')
def new_user():
    session.clear()
    return redirect(url_for('survey'))

if __name__ == '__main__':
    app.run(debug=True)