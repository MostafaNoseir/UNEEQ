import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from joblib import load
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Cached Loaders ---

@st.cache_data
def load_data():
    ratings = pd.read_csv('rating.csv')
    movies = pd.read_csv('movie.csv')
    tags = pd.read_csv('tag.csv')
    return ratings, movies, tags

@st.cache_resource
def load_trained_model():
    return load('svd_trained_model.joblib')

@st.cache_resource
def build_model_and_trainset(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data.build_full_trainset()

@st.cache_resource
def build_tfidf_content_model(movies, tags):
    # Preprocess genres and tags
    movies['processed_genres'] = movies['genres'].str.replace('|', ' ', regex=False)
    tags['tag'] = tags['tag'].astype(str)
    tag_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(tag_grouped, on='movieId', how='left')
    movies['text'] = movies['processed_genres'] + ' ' + movies['tag'].fillna('')

    # Build TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['text'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title'])
    return movies, cosine_sim, indices

# --- Load Cached Resources ---
ratings, movies, tags = load_data()
final_model = load_trained_model()
trainset = build_model_and_trainset(ratings)
movies, cosine_sim, indices = build_tfidf_content_model(movies, tags)
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))

# --- Recommendation Functions ---

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def recommend_for_user(user_id):
    try:
        inner_user_id = trainset.to_inner_uid(user_id)
    except ValueError:
        return None, "User ID not found."

    rated_items = set(j for (j, _) in trainset.ur[inner_user_id])
    anti_testset = [
        (user_id, trainset.to_raw_iid(i), 0.0)
        for i in trainset.all_items()
        if i not in rated_items
    ]
    predictions = final_model.test(anti_testset)
    top_n = get_top_n(predictions, n=10)
    return top_n.get(user_id, []), None

def content_recommendations(title, n=10):
    if title not in indices:
        return None, "Movie title not found."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres']].iloc[movie_indices], None

# --- Streamlit UI ---

st.title("🎬 Movie Recommender System")

tabs = st.tabs(["📊 Collaborative Filtering", "🎯 Content-Based Filtering"])

# --- Collaborative Filtering Tab ---
with tabs[0]:
    st.subheader("Top 10 Personalized Recommendations")

    valid_user_ids = ratings['userId'].unique()
    user_input = st.number_input("Enter your User ID",
                                 min_value=int(valid_user_ids.min()),
                                 max_value=int(valid_user_ids.max()),
                                 step=1)

    if st.button("Get User-Based Recommendations"):
        recs, error = recommend_for_user(user_input)
        if error:
            st.error(error)
        elif not recs:
            st.warning("No recommendations found.")
        else:
            st.success(f"Top 10 Recommendations for User {user_input}")
            for movie_id, score in recs:
                title = movie_id_to_title.get(int(movie_id), "Unknown Movie")
                st.write(f"✅ {title} (Predicted Rating: {round(score, 2)})")

# --- Content-Based Filtering Tab ---
with tabs[1]:
    st.subheader("Similar Movies Based on Genre/Tags")
    movie_title = st.selectbox("Choose a movie title", sorted(movies['title'].dropna().unique()))

    if st.button("Find Similar Movies"):
        recs, error = content_recommendations(movie_title, n=10)
        if error:
            st.error(error)
        elif recs is None or recs.empty:
            st.warning("No similar movies found.")
        else:
            st.success(f"Top 10 Movies Similar to '{movie_title}'")
            st.table(recs.reset_index(drop=True))