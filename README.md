# Music Recommender System

This project implements a music recommender system using collaborative filtering, with a focus on the Million Song Dataset. The system predicts and suggests songs to users based on their listening history and the preferences of similar users. It is designed to enhance user experience by delivering personalized music recommendations.

---

## Project Structure

```
.
├── code.ipynb                 # Main Jupyter Notebook (implementation and EDA)
├── research-paper.docx        # Research paper (background, methodology, results)
├── songs.csv                  # Merged and cleaned dataset (user-song interactions + metadata)
```

---

## Dataset Description

- **Source:** [Million Song Dataset](http://millionsongdataset.com/)
- **song_info:** Contains user_id, song_id, and listen_count (number of times a user played a song)
- **song_actual:** Contains song_id, title, release, artist_name, and year
- **Merged Dataset:** Combines the above, providing user-song interactions with song metadata

---

## Recommendation System Types

- **Content-Based Filtering:** Recommends songs similar to those a user liked in the past, based on song features (genre, artist, etc.)
- **Collaborative Filtering:** Recommends songs based on the preferences of similar users. This project focuses on collaborative filtering, specifically k-Nearest Neighbors (kNN) for user-user and item-item similarity.
- **Hybrid Models:** Combine collaborative and content-based approaches for improved accuracy and to address cold-start problems.

---

## Approach & Methodology

1. **Data Loading & Merging**
   - User-song interaction and song metadata are loaded and merged on `song_id`.
2. **Exploratory Data Analysis (EDA)**
   - Analyze most popular songs, artists, and user listening patterns.
   - Visualize listen counts and distribution of plays across users and songs.
3. **Data Preparation**
   - Create a user-item interaction matrix.
   - Filter out users and songs with very few interactions to focus on active users and popular tracks.
4. **Collaborative Filtering with kNN**
   - Build a sparse user-item matrix.
   - Use k-Nearest Neighbors to find similar users or items and generate recommendations.
5. **Evaluation**
   - Assess system performance using metrics such as precision, recall, and F1-score.
   - Discuss cold-start and scalability challenges, and propose hybrid or deep learning approaches for future improvements.

---

## How to Run

1. **Install Requirements**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
2. **Open the Notebook**
   - Launch `code.ipynb` in Jupyter Notebook or Google Colab.
3. **Run All Cells**
   - The notebook will:
     - Load and merge data
     - Perform EDA
     - Build and evaluate the collaborative filtering recommender

---

## Example: Key Code Snippet

```python
import pandas as pd
from scipy.sparse import csr_matrix
from recommeders.knn_recommender import Recommender

# Load and merge data
song_info = pd.read_csv('https://static.turi.com/datasets/millionsong/10000.txt', sep='\t', header=None)
song_info.columns = ['user_id', 'song_id', 'listen_count']
song_actual = pd.read_csv('https://static.turi.com/datasets/millionsong/song_data.csv')
song_actual.drop_duplicates(['song_id'], inplace=True)
songs = pd.merge(song_info, song_actual, on="song_id", how="left")

# Save for future use
songs.to_csv('songs.csv', index=False)

# Build user-item matrix for collaborative filtering
user_song_matrix = songs.pivot_table(index='user_id', columns='song_id', values='listen_count', fill_value=0)
sparse_matrix = csr_matrix(user_song_matrix.values)

# Example: Fit and use kNN recommender
recommender = Recommender()
recommender.fit(sparse_matrix)
```

---

## Results & Findings

- **Collaborative Filtering:** kNN-based collaborative filtering effectively recommends songs to active users based on similar user preferences.
- **Popular Songs & Artists:** The system identifies and recommends trending songs and artists.
- **Limitations:** Cold-start (new users/songs), data sparsity, and scalability are challenges. Hybrid models and deep learning (e.g., neural collaborative filtering, VAEs, RNNs) are discussed in the research paper for future improvement.

---

## Challenges & Solutions

- **Data Sparsity:** Addressed by filtering for active users/songs and using matrix factorization.
- **Cold Start:** Hybrid models and content-based features can help recommend for new users or songs.
- **Scalability:** Efficient algorithms (e.g., SVD, kNN) and sparse matrix representations are used for large-scale data.

---

## Ethical Considerations

- **Privacy:** User data is anonymized; recommendations are generated without exposing personal information.
- **Bias:** Regular auditing and diversity-promoting mechanisms are recommended to avoid filter bubbles and ensure fair recommendations.
- **Copyright:** Recommendations are based on legally available datasets; ensure compliance when deploying with proprietary music catalogs.

---

## References

- Million Song Dataset
- Research literature on collaborative filtering, content-based, hybrid, and deep learning recommendation systems (see `research-paper.docx` for full citations).

---

## Summary

This project demonstrates a collaborative filtering-based music recommender system using real-world data. It lays the foundation for more advanced, hybrid, and context-aware recommendation engines. For detailed methodology, results, and future work, refer to the included research paper.

---


Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/69564075/39ebda27-d1b3-4da2-bf74-8a9ce0861660/code.ipynb
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/69564075/c29f0fed-955b-49b3-8430-70360c8526f0/research-paper.docx
[3] https://github.com/sathishprasad/Music-Recommendation-System
[4] https://github.com/EddieAmaitum/Music-Recommender-System
[5] https://www.kaggle.com/code/favourdi/music-recommendation-system-mit-capstone
[6] https://www.scribd.com/document/808866031/Music-recommendation-system-Mini-project
[7] https://www.politesi.polimi.it/retrieve/a81cb059-877a-616b-e053-1605fe0a889a/thesis-mrs-carlos-alvarez.pdf
[8] http://ir.juit.ac.in:8080/jspui/bitstream/123456789/8276/1/Music%20Recommendation%20System.pdf
[9] https://colab.research.google.com/github/Gurobi/modeling-examples/blob/master/music_recommendation/music_recommendation.ipynb
[10] https://www.slideshare.net/slideshow/music-recommendation-system-project-pptpptx/262513442

