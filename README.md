# Book Recommendation Project

## Project Overview

This project builds a **Book Recommendation System** using data collection, exploratory data analysis, NLP, unsupervised learning, and Streamlit deployment.

The goal is to help users discover books similar to the ones they like. The final application allows users to explore books, view clusters, and receive personalized recommendations based on textual similarity, rating, popularity, and cluster information.

---

## Presentation

The final project presentation is available here:

[View the Canva Presentation](https://canva.link/2mmxvy1gqg0s7h2)

---

## Data Sources

The final dataset combines two sources:

| Source | Method | Purpose | Link |
|---|---|---|---|
| Goodreads Popular by Year | Web scraping | Top 10 Books from 1975 to 2024 | [Goodreads Popular Books by Date](https://www.goodreads.com/book/popular_by_date/2019) |
| Open Library | API | 10 Books from 1900 to 1975 | [Open Library API](https://openlibrary.org/developers/api) |

The combined dataset covers books from **1900 to 2024**.

---

## Dataset Features

The final dataset includes:

- Title
- Author
- Publication/source year
- Average rating
- Number of ratings
- Number of reviews
- Book description
- Genres
- Number of pages
- Book URL
- Image URL
- Data source
- Cluster label
- PCA coordinates

---

## Clustering Analysis

K-Means clustering was used to group books into broad reading profiles.

### Features Used for Clustering

Numerical features:

- Average rating
- Ratings count
- Reviews count
- Pages
- Source year

### Selected Number of Clusters

The final model used **6 clusters**.

The choice was based on:

- Elbow method
- Silhouette score
- Interpretability of the resulting groups

### Final Cluster Labels

| Cluster | Label |
|---|---|
| 0 | Historical Fiction |
| 1 | General Fiction |
| 2 | Mystery & Thriller |
| 3 | Children’s Books |
| 4 | Romance & YA |
| 5 | Nonfiction |

---

## PCA Analysis

PCA was applied to reduce the clustering feature space into two dimensions.

The first two components explained approximately **23% of the total variance**.

---

## NLP Recommendation System

The recommendation model is content-based.

It uses:

- Title
- Author
- Genres
- Description
- Cluster label

These text features are cleaned and transformed using **TF-IDF**.

Then, **cosine similarity** is used to compare each book with every other book.

---

## Weighted Recommendation Score

The final recommendation function combines three components:

```text
Final Score = 70% text similarity + 20% average rating + 10% popularity
```

| Component | Weight | Meaning |
|---|---:|---|
| Text similarity | 70% | How similar the book is based on text and metadata |
| Average rating | 20% | How well-rated the book is |
| Popularity | 10% | How many users rated the book |

This makes the recommender prioritize content similarity while still giving a small boost to well-rated and popular books.

---

## Streamlit App

The final app was built with **Streamlit**.

### App Pages

1. **Home**
   - Project overview
   - Dataset KPIs
   - Data source distribution

2. **Explore Books**
   - Filter books by source, cluster, rating, year, and genre
   - View rating distribution

3. **Recommendation System**
   - Select a book
   - Get similar book recommendations
   - View similarity score and final score

4. **Clusters & PCA**
   - View cluster summary
   - Explore PCA visualization
   - See number of books per cluster

5. **About**
   - Data, cosine similarity, and recommendation scoring

---

## Project Structure

```text
Book_Project/
│
├── data/
│   ├── raw/
│   │   ├── API_Books.csv
│   │   ├── goodreads_book_links_by_year.csv
│   │   └── goodreads_books_detailed1.csv
│   │
│   ├── API_Books_clean.csv
│   ├── Books_Clusters_PCA.csv
│   ├── GoodBooks_clean.csv
│   └── join_API_Goodreads.csv
│
├── notebooks/
│   ├── data_0_API.ipynb
│   ├── data_0_book_scrapping.ipynb
│   ├── data_1_EDA.ipynb
│   ├── data_2_PCA_Cluster.ipynb
│   └── data_3_NLP.ipynb
│
├── app.py
└── README.md

```

---

## How to Run the App

```bash
python -m streamlit run app.py
```

If Streamlit is not installed:

```bash
pip install streamlit
```

---

## Limitations

- Goodreads and Open Library provide different types of metadata.
- Some variables are influenced by the data source.
- Open Library descriptions are less detailed than Goodreads descriptions.
- PCA explains only part of the total variance because the original feature space is high-dimensional.
- The recommendation system is content-based and does not use individual user preferences or user ratings.

---

## Future Improvements

- Add user-based collaborative filtering if user rating data becomes available.
- Add more book sources or APIs.
- Improve genre standardization.
- Add sentiment or mood detection from reviews.
- Add user feedback in the Streamlit app.
- Deploy the app online using Streamlit Community Cloud.

---

## Author

Developed by **Bryan Calderon**  
For the **Ironhack 2026 Final Project**  
Germany