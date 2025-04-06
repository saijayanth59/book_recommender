# Book Recommender System

This Streamlit application provides a book recommendation system with various models. It utilizes pre-processed book data and user ratings to generate personalized recommendations.

## Features

- **Simple Recommender:** Recommends books based on popularity and average ratings.
- **Content-Based Filtering:** Recommends books similar to a selected book based on content similarity.
- **Content-Based Filtering+:** Improves content-based recommendations by filtering out low-rated books.
- **Collaborative Filtering:** Recommends books based on user preferences using a pre-trained Singular Value Decomposition (SVD) model.

## Setup and Run Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/saijayanth59/book_recommender.git
    cd book_recommender
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run streamlit**
    ```bash
    streamlit run main.py
    ```
