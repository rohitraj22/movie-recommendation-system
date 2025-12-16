# Movie Recommendation System using Matrix Factorization (SVD)

## 1. Project Overview
This project builds a collaborative filtering recommendation system for movies using the **MovieLens (ml-32m)** dataset. By utilizing **Singular Value Decomposition (SVD)** from the `surprise` library, the system predicts user ratings and generates personalized Top-N movie recommendations.

The final model was trained on **32 million user-item interactions**, achieving an RMSE of **0.8259**, indicating strong predictive accuracy.

### Group 8 Members
* **220343** Devansh Chaudhary
* **220435** Harshika Agrawal
* **220475** Jatin Madan
* **220505** Kartikeya Katiyar
* **220915** Rohit Raj

---

## 2. Dataset & Preprocessing

The project utilizes the **ml-32m** dataset, which was processed into a single analytical file named `cleaned_movies.csv`.

### Data Cleaning Steps
1.  **Merging:** The `movies.csv` and `ratings.csv` files were merged on `movieId`.
2.  **Handling Missing Data:** 3,153 rows containing null values (movies with no ratings) were removed to create a dense user-item matrix.
3.  **Feature Engineering:**
    * **Genres:** The pipe-separated `genres` string was converted into One-Hot Encoded binary columns (e.g., `Action`, `Comedy`).
    * **Unknown Label:** The category `(no genres listed)` was renamed to `Unknown`.
    * **Timestamps:** Unix timestamps were converted to datetime objects to extract a `rating_year` feature.

**Final Dataset Shape:** 32,000,204 rows × 26 columns.

---

## 3. Exploratory Data Analysis (EDA)



Key insights from the analysis include:
* **Positivity Bias:** The distribution of ratings is heavily skewed towards positive scores, with **4.0** being the most common rating.
* **Content Popularity:** The most rated movie is *"The Shawshank Redemption (1994)"* (~100k ratings), followed by *"Forrest Gump"* and *"Pulp Fiction"*.
* **Genre Landscape:** **Drama** is the most prevalent genre (13M+ ratings), followed by **Comedy** and **Action**.

---

## 4. Model Architecture

The recommendation engine uses Matrix Factorization via SVD.

### Final Model Hyperparameters
Based on validation performance, the final model was trained with the following parameters:
* **Algorithm:** SVD (Singular Value Decomposition)
* **Latent Factors (`n_factors`):** 50
* **Epochs (`n_epochs`):** 10
* **Learning Rate (`lr_all`):** 0.005
* **Regularization (`reg_all`):** 0.04
* **Train/Test Split:** 80% Train / 20% Test.

---

## 5. Performance Evaluation

The model was evaluated using both error-based metrics (accuracy) and ranking metrics (recommendation quality).

| Metric | Score | Description |
| :--- | :--- | :--- |
| **RMSE** | **0.8259** | Root Mean Squared Error (Standard deviation of prediction error). |
| **MAE** | **0.6243** | Mean Absolute Error (Average absolute difference in ratings). |
| **Precision@10** | **0.6625** | Percentage of recommended items that are relevant. |
| **Recall@10** | **0.6679** | Percentage of relevant items that were successfully recommended. |

---

## 6. Installation & Requirements

The project requires Python and the following libraries:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-surprise`
