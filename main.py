import numpy as np
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import dump


@st.cache_data
def read_book_data():
    return pd.read_csv('data/books_cleaned.csv')


@st.cache_data
def read_ratings_data():
    return pd.read_csv("data/ratings.csv")


@st.cache_resource
def load_model():
    _, model = dump.load('./models/dump_tuning')
    return model


@st.cache_data
def content(books):
    books['content'] = (pd.Series(books[['authors', 'title', 'genres', 'description']]
                                  .fillna('')
                                  .values.tolist()
                                  ).str.join(' '))

    tf_content = TfidfVectorizer(analyzer='word', ngram_range=(
        1, 2), min_df=1, stop_words='english')

    tfidf_matrix = tf_content.fit_transform(books['content'])
    cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
    index = pd.Series(books.index, index=books['title'])

    return cosine, index


def book_read(books, ratings_data, user_id):
    """Take user_id and return list of book that user has read"""
    books_list = list(books['book_id'])
    book_read_list = list(
        ratings_data['book_id'][ratings_data['user_id'] == user_id])
    return books_list, book_read_list


def get_recommendation_svd(books, ratings_data, user_id, n=5):
    """Give n recommendations to user_id using pre-trained SVD model."""

    svd = load_model()

    all_books, user_books = book_read(books, ratings_data, user_id)
    next_books = [book for book in all_books if book not in user_books]

    if n <= len(next_books):
        ratings = []
        for book in next_books:
            est = svd.predict(user_id, book).est
            ratings.append((book, est))
        ratings = sorted(ratings, key=lambda x: x[1], reverse=True)
        book_ids = [id_ for id_, _ in ratings[:n]]
        return books[books.book_id.isin(book_ids)][['book_id', 'title', 'authors', 'average_rating', 'ratings_count']]
    else:
        st.warning(
            "⚠️ Please reduce your recommendation request — too few unseen books.")
        return pd.DataFrame()


def simple_recommender(books, n=5):
    v = books['ratings_count']
    m = books['ratings_count'].quantile(0.95)
    R = books['average_rating']
    C = books['average_rating'].median()
    books['score'] = (v / (v + m) * R) + (m / (m + v) * C)
    qualified = books.sort_values('score', ascending=False)
    return qualified[['book_id', 'title', 'authors', 'score']].head(n)


def content_recommendation(books, title, n=5):
    cosine_sim, indices = content(books)
    idx = indices[title]
    sim_scores = sorted(
        list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n + 1]
    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices][['book_id', 'title', 'authors', 'average_rating', 'ratings_count']]


def improved_recommendation(books, title, n=5):
    cosine_sim, indices = content(books)
    idx = indices[title]
    sim_scores = sorted(
        list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:26]
    book_indices = [i[0] for i in sim_scores]
    books2 = books.iloc[book_indices][['book_id', 'title',
                                       'authors', 'average_rating', 'ratings_count']]

    v = books2['ratings_count']
    m = books2['ratings_count'].quantile(0.75)
    R = books2['average_rating']
    C = books2['average_rating'].median()
    books2['weighted_rating'] = (v / (v + m) * R) + (m / (m + v) * C)

    return books2[books2['ratings_count'] >= m].sort_values('weighted_rating', ascending=False).head(n)


def main():
    st.set_page_config(page_title="Book Recommender",
                       page_icon="📔", layout="centered")

    st.title("📚 Book Recommender")
    with st.expander("ℹ️ See explanation"):
        st.write("""
        **Models Available:**
        1. **Simple Recommender** - Recommends books based on popularity and ratings.
        2. **Content-Based Filtering** - Recommends books similar to a selected one.
        3. **Content-Based Filtering+** - Filters low-rated books for better recommendations.
        4. **Collaborative Filtering** - Recommends books based on user preferences.
             In this model, you can explore what system recommend to certain user registered in the dataset. 
             The recommendation for new user is not available at the moment. You can pick user ID and specify number of 
             books to recommend. This model will show recommendation for that particular user.
             
             PS: It takes approximately 2 minutes to get recommendation on collaborative filtering.
        """)

    books = read_book_data().copy()

    model, book_num = st.columns((2, 1))
    selected_model = model.selectbox('Select Model', [
                                     'Simple Recommender', 'Content Based Filtering', 'Content Based Filtering+', 'Collaborative Filtering'])
    selected_book_num = book_num.selectbox(
        'Number of Books', [5, 10, 15, 20, 25])

    if selected_model == 'Simple Recommender':
        if st.button('🔍 Recommend'):
            try:
                recs = simple_recommender(books, selected_book_num)
                st.dataframe(recs)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif selected_model == 'Collaborative Filtering':
        ratings_data = read_ratings_data()
        user_id_picked = st.number_input(
            label="User ID:", min_value=1, max_value=60000)
        if st.button('Recommend'):
            if user_id_picked in ratings_data["user_id"].unique():
                with st.spinner('Getting recommendation for you...'):
                    recs = get_recommendation_svd(books=books,
                                                  ratings_data=ratings_data,
                                                  user_id=user_id_picked,
                                                  n=selected_book_num)
                st.write(recs)
            else:
                st.write('You have entered an invalid User ID')

    else:
        options = np.concatenate(([''], books["title"].unique()))
        book_title = st.selectbox('Pick a Book', options, 0)

        if book_title == '':
            st.warning('⚠️ Please select a book first.')
            return

        if st.button('🔍 Recommend'):
            try:
                if selected_model == 'Content Based Filtering':
                    recs = content_recommendation(
                        books, book_title, selected_book_num)
                else:
                    recs = improved_recommendation(
                        books, book_title, selected_book_num)
                st.dataframe(recs)
            except Exception as e:
                st.error(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
