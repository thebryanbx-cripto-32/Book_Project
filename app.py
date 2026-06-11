import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Book Recommender App",
    layout="wide")


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/Books_Clusters_PCA.csv")

    required_columns = [
        "title", "author", "average_rating", "ratings_count",
        "reviews_count", "description", "genres", "pages",
        "source_year", "source", "cluster", "cluster_label",
        "pca_1", "pca_2", "image_url", "book_url"]

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["author"] = df["author"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("").astype(str)
    df["cluster_label"] = df["cluster_label"].fillna("Unknown").astype(str)
    df["source"] = df["source"].fillna("Unknown").astype(str)
    df["image_url"] = df["image_url"].fillna("").astype(str)
    df["book_url"] = df["book_url"].fillna("").astype(str)

    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0)
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0).astype(int)
    df["reviews_count"] = pd.to_numeric(df["reviews_count"], errors="coerce").fillna(0).astype(int)
    df["pages"] = pd.to_numeric(df["pages"], errors="coerce").fillna(df["pages"].median())
    df["source_year"] = pd.to_numeric(df["source_year"], errors="coerce").fillna(0).astype(int)

    return df


df = load_data()


# --------------------------------------------------
# NLP helper functions
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def build_similarity_model(df):
    df_model = df.copy()

    df_model["combined_text"] = (
        df_model["title"].fillna("") + " " +
        df_model["author"].fillna("") + " " +
        df_model["genres"].fillna("") + " " +
        df_model["description"].fillna("") + " " +
        df_model["cluster_label"].fillna("")
    )

    df_model["clean_text"] = df_model["combined_text"].apply(clean_text)

    custom_stop_words = list(ENGLISH_STOP_WORDS.union([
        "book", "books", "story", "stories", "novel", "novels",
        "fiction", "general", "literature", "english",
        "edition", "author", "read", "reader", "readers"
    ]))

    tfidf = TfidfVectorizer(
        stop_words=custom_stop_words,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85
    )

    tfidf_matrix = tfidf.fit_transform(df_model["clean_text"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    return df_model, cosine_sim


df_model, cosine_sim = build_similarity_model(df)


def recommend_books_weighted(book_title, df, cosine_sim, n=5):
    matches = df[df["title"] == book_title]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = [(i, score) for i, score in similarity_scores if i != idx]

    max_ratings = np.log1p(df["ratings_count"].max())

    recommendations = []

    for i, similarity in similarity_scores:
        rating_scaled = df.loc[i, "average_rating"] / 5

        if max_ratings > 0:
            popularity_scaled = np.log1p(df.loc[i, "ratings_count"]) / max_ratings
        else:
            popularity_scaled = 0

        same_cluster_bonus = 0.05 if df.loc[i, "cluster_label"] == df.loc[idx, "cluster_label"] else 0

        final_score = (
            0.70 * similarity +
            0.20 * rating_scaled +
            0.10 * popularity_scaled +
            same_cluster_bonus
        )

        recommendations.append((i, similarity, final_score))

    recommendations = sorted(
        recommendations,
        key=lambda x: x[2],
        reverse=True
    )

    top_books = recommendations[:n]
    book_indices = [i[0] for i in top_books]

    results = df.iloc[book_indices][[
        "title", "author", "source_year", "average_rating",
        "ratings_count", "genres", "cluster_label",
        "description", "image_url", "book_url", "source"
    ]].copy()

    results["similarity_score"] = [i[1] for i in top_books]
    results["final_score"] = [i[2] for i in top_books]

    return results


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.title("📚 Book Recommender")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Explore Books",
        "Recommendation System",
        "Clusters & PCA",
        "About the Model"
    ]
)


# --------------------------------------------------
# Page 1: Home
# --------------------------------------------------

if page == "Home":
    st.title("Book Recommendation System")

    st.markdown(
        """
        This app recommends books using Goodreads and Open Library data.
        It combines exploratory data analysis, NLP, clustering, PCA, and a content-based recommendation system.
        """
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Books", f"{df.shape[0]:,}")
    col2.metric("Unique Authors", f"{df['author'].nunique():,}")
    col3.metric("Year Range", f"{df['source_year'].min()} - {df['source_year'].max()}")
    col4.metric("Avg Rating", f"{df['average_rating'].mean():.2f}")
    col5.metric("Clusters", f"{df['cluster_label'].nunique()}")

    st.subheader("Dataset Sources")

    source_counts = df["source"].value_counts().reset_index()
    source_counts.columns = ["source", "book_count"]

    fig = px.bar(
        source_counts,
        x="source",
        y="book_count",
        title="Books by Data Source",
        labels={"source": "Source", "book_count": "Number of Books"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Project Workflow Summary")

    st.markdown(
        """
        1. **Data collection**: Goodreads scraping and Open Library API  
        2. **Cleaning and EDA**: missing values, duplicates, genres, ratings, years  
        3. **Clustering**: K-Means using numerical features and encoded genres  
        4. **PCA**: 2D visualization of book clusters  
        5. **NLP recommendation**: TF-IDF and cosine similarity  
        """
    )

    st.markdown("---")

    st.markdown(
        """
        Developed by **Bryan Calderon**  
        For the **Ironhack Final Project**  
        **Germany 2026**
        """
    )


# --------------------------------------------------
# Page 2: Explore Books
# --------------------------------------------------

elif page == "Explore Books":
    st.title("Explore Books")

    st.sidebar.subheader("Filters")

    selected_source = st.sidebar.multiselect(
        "Source",
        options=sorted(df["source"].unique()),
        default=sorted(df["source"].unique())
    )

    selected_clusters = st.sidebar.multiselect(
        "Cluster",
        options=sorted(df["cluster_label"].unique()),
        default=sorted(df["cluster_label"].unique())
    )

    min_rating = st.sidebar.slider(
        "Minimum Rating",
        min_value=float(df["average_rating"].min()),
        max_value=float(df["average_rating"].max()),
        value=float(df["average_rating"].min()),
        step=0.1
    )

    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df["source_year"].min()),
        max_value=int(df["source_year"].max()),
        value=(int(df["source_year"].min()), int(df["source_year"].max()))
    )

    genre_search = st.sidebar.text_input("Search genre keyword")

    filtered_df = df[
        (df["source"].isin(selected_source)) &
        (df["cluster_label"].isin(selected_clusters)) &
        (df["average_rating"] >= min_rating) &
        (df["source_year"].between(year_range[0], year_range[1]))
    ].copy()

    if genre_search:
        filtered_df = filtered_df[
            filtered_df["genres"].str.lower().str.contains(
                genre_search.lower(),
                na=False
            )
        ]

    st.write(f"Showing **{filtered_df.shape[0]}** books")

      # Adding better labels for the table
    display_df = filtered_df[[
        "title",
        "author", "source_year", "average_rating", "ratings_count",
        "reviews_count", "genres", "cluster_label","source"]].copy()

    display_df = display_df.rename(columns={
        "title": "Title",
        "author": "Author",
        "source_year": "Publication Year",
        "average_rating": "Average Rating",
        "ratings_count": "Number of Ratings",
        "reviews_count": "Number of Reviews",
        "genres": "Genres",
        "cluster_label": "Book Cluster",
         "source": "Data Source"
    })

    st.dataframe(
    display_df.sort_values("Average Rating", ascending=False),
    use_container_width=True
    )

    st.subheader("Rating Distribution")

    fig = px.histogram(
        filtered_df,
        x="average_rating",
        nbins=25,
        title="Distribution of Average Ratings" 
    )

    fig.update_layout(xaxis_title = "Average Rating",
        yaxis_title = "Number of Books")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown(
        """
        Developed by **Bryan Calderon**  
        For the **Ironhack Final Project**  
        **Germany 2026**
        """
    )

# --------------------------------------------------
# Page 3: Recommendation System
# --------------------------------------------------

elif page == "Recommendation System":
    st.title("Book Recommendation System")

    st.markdown(
        """
        Select a book and the app will recommend similar books using TF-IDF, cosine similarity,
        rating, popularity, and cluster information.
        """
    )

    selected_book = st.selectbox(
        "Choose a book",
        options=sorted(df_model["title"].unique())
    )

    n_recommendations = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    selected_book_info = df_model[df_model["title"] == selected_book].iloc[0]

    st.subheader("Selected Book")

    col1, col2 = st.columns([1, 3])

    with col1:
        if selected_book_info["image_url"]:
            st.image(selected_book_info["image_url"], width=160)

    with col2:
        st.markdown(f"### {selected_book_info['title']}")
        st.write(f"**Author:** {selected_book_info['author']}")
        st.write(f"**Year:** {selected_book_info['source_year']}")
        st.write(f"**Rating:** {selected_book_info['average_rating']:.2f}")
        st.write(f"**Cluster:** {selected_book_info['cluster_label']}")
        st.write(f"**Genres:** {selected_book_info['genres']}")

        if selected_book_info["book_url"]:
            st.markdown(f"[Open book page]({selected_book_info['book_url']})")

    if st.button("Recommend Books"):
        recommendations = recommend_books_weighted(
            selected_book,
            df_model,
            cosine_sim,
            n=n_recommendations
        )

        st.subheader("Recommended Books")

        for _, row in recommendations.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])

                with col1:
                    if row["image_url"]:
                        st.image(row["image_url"], width=120)

                with col2:
                    st.markdown(f"### {row['title']}")
                    st.write(f"**Author:** {row['author']}")
                    st.write(f"**Year:** {row['source_year']}")
                    st.write(f"**Rating:** {row['average_rating']:.2f}")
                    st.write(f"**Ratings Count:** {row['ratings_count']:,}")
                    st.write(f"**Cluster:** {row['cluster_label']}")
                    st.write(f"**Similarity Score:** {row['similarity_score']:.3f}")
                    st.write(f"**Final Score:** {row['final_score']:.3f}")
                    st.write(f"**Genres:** {row['genres']}")

                    if row["book_url"]:
                        st.markdown(f"[Open book page]({row['book_url']})")

                st.divider()


# --------------------------------------------------
# Page 4: Clusters & PCA
# --------------------------------------------------

elif page == "Clusters & PCA":
    st.title("Clusters & PCA Visualization")

    st.markdown(
        """
        Books were grouped using K-Means clustering based on numerical features and encoded genres.
        PCA was used to reduce the feature space to two dimensions for visualization.
        """
    )

    st.subheader("Cluster Summary")

    cluster_summary = (
        df.groupby("cluster_label")
        .agg(
            book_count=("title", "count"),
            avg_rating=("average_rating", "mean")
        )
        .reset_index()
    )

    st.dataframe(cluster_summary.round(2), use_container_width=True)

    st.subheader("PCA Visualization")

    fig = px.scatter(
        df,
        x="pca_1",
        y="pca_2",
        color="cluster_label",
        hover_data=["title", "author", "source_year", "average_rating", "source"],
        title="PCA Visualization of Book Clusters",
        labels={
            "pca_1": "Principal Component 1",
            "pca_2": "Principal Component 2",
            "cluster_label": "Cluster"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Books by Cluster")

    cluster_counts = df["cluster_label"].value_counts().reset_index()
    cluster_counts.columns = ["cluster_label", "book_count"]

    fig2 = px.bar(
        cluster_counts,
        x="book_count",
        y="cluster_label",
        orientation="h",
        title="Number of Books per Cluster",
        labels={"book_count": "Number of Books", "cluster_label": "Cluster"}
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.markdown(
        """
       

        Developed by **Bryan Calderon**  
        For the **Ironhack Final Project**  
        **Germany 2026**
     """
     )


# --------------------------------------------------
# Page 5: About the Model
# --------------------------------------------------

elif page == "About the Model":
    st.title("About the Model")

    st.subheader("Data")

    st.markdown(
        """
        The project combines two data sources:

        - **Goodreads Popular by Year**: scraped data for more recent popular books.
        - **Open Library API**: historical book metadata for older books.

        The final dataset includes book titles, authors, ratings, descriptions, genres, years, pages, cover images, and source information.
        """
    )

    st.subheader("Recommendation System")

    st.markdown(
        """
        The recommendation system is content-based.

        It combines:

        - Title
        - Author
        - Genres
        - Description
        - Cluster label

        These text features are transformed with **TF-IDF**.
        Then, **cosine similarity** measures how similar each book is to every other book.
        The final recommendation score also includes rating, popularity, and a small same-cluster bonus.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Project Information

        Developed by **Bryan Calderon**  
        For the **Ironhack Final Project**  
        **Germany 2026**
     """
     )

# run in powershéll "python -m streamlit run app.py"

