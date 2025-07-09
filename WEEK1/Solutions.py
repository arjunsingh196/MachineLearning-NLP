import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Problem 1: Array Operations
def generate_random_matrix(rows=5, cols=4, low=1, high=50):
    return np.random.randint(low, high, size=(rows, cols))

def extract_anti_diagonal(matrix):
    return np.array([matrix[i, matrix.shape[1] - i - 1] for i in range(min(matrix.shape))])

def get_row_maxima(matrix):
    return np.max(matrix, axis=1)

def calculate_matrix_mean(matrix):
    return np.mean(matrix)

def filter_below_mean(matrix):
    return matrix[matrix <= calculate_matrix_mean(matrix)]

def traverse_boundary(matrix):
    rows, cols = matrix.shape
    top_row = matrix[0, :].tolist()
    bottom_row = matrix[rows - 1, :][::-1].tolist()
    right_col = matrix[1:rows - 1, cols - 1].tolist()
    left_col = matrix[1:rows - 1, 0][::-1].tolist()
    return top_row + right_col + bottom_row + left_col

# Problem 2: Array Processing
def generate_uniform_array(size=20, low=0, high=10):
    return np.random.uniform(low, high, size=size)

def round_array(array, decimals=2):
    return np.round(array, decimals=decimals)

def get_array_stats(array):
    return {
        "max": np.max(array),
        "min": np.min(array),
        "median": np.median(array)
    }

def square_elements_below_threshold(array, threshold=5):
    return np.where(array < threshold, array**2, array)

def alternate_sort(array):
    sorted_array = np.sort(array)
    result = []
    left, right = 0, len(sorted_array) - 1
    while left < right:
        result.extend([sorted_array[left], sorted_array[right]])
        left += 1
        right -= 1
    if left == right:
        result.append(sorted_array[left])
    return np.array(result)

# Problem 3: DataFrame Operations
def create_student_dataframe(size=10):
    return pd.DataFrame({
        "student_name": [f"S{i+1}" for i in range(size)],
        "subject": ["Math", "CS", "Physics", "Chem", "Math", "CS", "Chem", "CS", "Physics", "Math"],
        "score": np.random.randint(50, 100, size=size).tolist(),
        "grade": [""] * size
    })

def assign_grades(df):
    conditions = [
        (df["score"] >= 90),
        (df["score"] >= 80) & (df["score"] < 90),
        (df["score"] >= 70) & (df["score"] < 80),
        (df["score"] >= 60) & (df["score"] < 70),
        (df["score"] < 60)
    ]
    grades = ["A", "B", "C", "D", "F"]
    df["grade"] = np.select(conditions, grades, default="F")
    return df

def sort_by_score(df):
    return df.sort_values(by="score", ascending=False).reset_index(drop=True)

def calculate_subject_averages(df):
    return df.groupby("subject", as_index=False)["score"].mean()

def filter_passing_grades(df):
    return df[df["grade"].isin(["A", "B"])]

# Problem 4: Movie Review Sentiment Analysis
def create_movie_reviews_dataframe():
    positive_templates = [
        "Absolutely loved it!", "A fantastic experience from start to finish.",
        "Great acting and brilliant storyline.", "One of the best movies I've seen this year.",
        "Incredible direction and cinematography.", "A masterpiece. Would highly recommend.",
        "Emotionally moving and beautifully shot.", "Engaging, entertaining, and very well made.",
        "Excellent pacing and character development.", "An unforgettable cinematic journey.",
        "Loved every minute of it.", "The plot was thrilling and unpredictable.",
        "The cast did a phenomenal job.", "A perfect blend of drama and action.",
        "Hilarious and heartfelt at the same time.", "Simply amazing. I'd watch it again!",
        "Beautifully executed.", "A must-watch for everyone.",
        "Outstanding performance by the lead actor.", "This movie exceeded my expectations.",
        "A solid 10 out of 10!", "Top-notch production value.", "A delight to watch.",
        "The emotional scenes hit hard.", "Visually stunning and thought-provoking."
    ]
    negative_templates = [
        "Terribly disappointing.", "A complete waste of time.", "Poor acting and weak storyline.",
        "I expected much more from this movie.", "Failed to impress in any aspect.",
        "The plot was slow and boring.", "Couldn’t finish it. It was that bad.",
        "Flat characters and meaningless dialogue.", "Not worth the hype.",
        "Disjointed scenes and lazy writing.", "Predictable and dull.",
        "Felt like a bad TV episode.", "Unconvincing performances.",
        "The script was all over the place.", "I kept waiting for it to get better, it didn’t.",
        "Nothing memorable at all.", "Sloppy editing ruined it.", "Forgettable and bland.",
        "Regret watching this.", "It dragged on forever.", "Bad CGI and worse acting.",
        "Totally underwhelming.", "No chemistry between the leads.",
        "Tried to be deep, but failed.", "One of the worst movies I’ve seen."
    ]
    positive_reviews = random.choices(positive_templates, k=50)
    negative_reviews = random.choices(negative_templates, k=50)
    reviews = positive_reviews + negative_reviews
    sentiments = ["positive"] * 50 + ["negative"] * 50
    combined = list(zip(reviews, sentiments))
    random.shuffle(combined)
    return pd.DataFrame(combined, columns=["review", "sentiment"])

def train_sentiment_model(reviews_df):
    X = reviews_df["review"]
    y = reviews_df["sentiment"]
    vectorizer = CountVectorizer(max_features=500, stop_words="english")
    X_transformed = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=1)
    model = Pipeline([("classifier", MultinomialNB())])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    print(f"Accuracy: {accuracy*100}%")
    return model, vectorizer

def predict_movie_review(model, vectorizer, review_text):
    transformed_text = vectorizer.transform([review_text])
    return model.predict(transformed_text)[0]

# Problem 5: Product Review Sentiment Analysis
def create_product_reviews_dataframe():
    good_phrases = [
        "Excellent product", "Very satisfied", "Loved it", "Highly recommended", "Amazing experience",
        "Works perfectly", "Great value", "Top-notch quality", "Five stars", "Will buy again",
        "Superb performance", "Exceeded expectations", "Fantastic!", "Very happy with it", "Best purchase ever",
        "Just what I needed", "Reliable and efficient", "Great customer service", "User-friendly", "Awesome build quality",
        "Fast delivery", "Perfect fit", "Exactly as described", "Good packaging", "Affordable and good",
        "Top quality", "Smooth operation", "Beautiful design", "Super useful", "Nothing to complain about",
        "Very intuitive", "Product met all my needs", "Sturdy and well-made", "Looks great", "Easy to set up",
        "Amazing durability", "Really impressed", "Battery lasts long", "Stylish and practical", "Very effective",
        "Performs better than expected", "Nice and compact", "Pleasant experience", "Just perfect", "Efficient and sleek",
        "Feels premium", "Simply awesome", "Love the features", "Flawless", "Beyond expectations"
    ]
    bad_phrases = [
        "Terrible experience", "Very disappointed", "Broke after one use", "Would not recommend", "Poor quality",
        "Didn’t work", "Not worth the money", "Worst product", "One star", "Never buying again",
        "Slow delivery", "Not as described", "Cheap material", "Waste of money", "Stopped working soon",
        "Hard to use", "Poor customer service", "Overpriced", "Useless", "Looks different from photo",
        "Too complicated", "No instructions included", "Bad packaging", "Product arrived damaged", "Low durability",
        "Difficult to install", "Does not do what it claims", "Unreliable", "Bad design", "Too noisy",
        "Missing parts", "Doesn’t fit properly", "Feels flimsy", "Very slow", "Buttons don’t work",
        "Battery drains fast", "Not intuitive", "Disappointed with the quality", "Looks cheap", "Very annoying",
        "Uncomfortable", "Not user-friendly", "Regret buying it", "Barely works", "Returned it immediately",
        "Overheats quickly", "Feels like a toy", "No value for money", "Subpar build", "Bad experience"
    ]
    random.shuffle(good_phrases)
    random.shuffle(bad_phrases)
    reviews = good_phrases[:50] + bad_phrases[:50]
    labels = ["good"] * 50 + ["bad"] * 50
    df = pd.DataFrame({"review": reviews, "sentiment": labels})
    return df.sample(frac=1).reset_index(drop=True)

def train_product_review_model(reviews_df):
    X = reviews_df["review"]
    y = reviews_df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    model = Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=300, stop_words="english")),
        ("classifier", LogisticRegression(max_iter=200))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Recall Score: {recall_score(y_test, predictions, pos_label='good')}")
    print(f"Precision Score: {precision_score(y_test, predictions, pos_label='good')}")
    print(f"F1 Score: {f1_score(y_test, predictions, pos_label='good')}")
    print(classification_report(y_test, predictions))
    return model

# Execution
if __name__ == "__main__":
    # Problem 1
    matrix = generate_random_matrix()
    print("Random Matrix:\n", matrix)
    print("Anti-diagonal:", extract_anti_diagonal(matrix))
    print("Row Maxima:", get_row_maxima(matrix))
    print("Matrix Mean:", calculate_matrix_mean(matrix))
    print("Elements <= Mean:", filter_below_mean(matrix))
    sample_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Sample Matrix:\n", sample_matrix)
    print("Boundary Traversal:", traverse_boundary(sample_matrix))

    # Problem 2
    array = generate_uniform_array()
    print("Random Array:", array)
    rounded_array = round_array(array)
    print("Rounded Array:", rounded_array)
    stats = get_array_stats(rounded_array)
    print(f"Maximum: {stats['max']}, Minimum: {stats['min']}, Median: {stats['median']}")
    print("Squared Elements < 5:", square_elements_below_threshold(rounded_array))
    sample_array = np.array([1, 6, 3, 9, 10, 12, 5, 4])
    print("Alternate Sorted Array:", alternate_sort(sample_array))

    # Problem 3
    student_df = create_student_dataframe()
    print("Student DataFrame:")
    display(student_df)
    student_df = assign_grades(student_df)
    print("Graded DataFrame:")
    display(student_df)
    sorted_df = sort_by_score(student_df)
    print("Sorted DataFrame:")
    display(sorted_df)
    print("Subject Averages:")
    display(calculate_subject_averages(sorted_df))
    print("Scores Array:", np.array(sorted_df["score"]))
    print("Passing Grades DataFrame:")
    display(filter_passing_grades(sorted_df))

    # Problem 4
    movie_reviews_df = create_movie_reviews_dataframe()
    print("Movie Reviews DataFrame:")
    display(movie_reviews_df)
    sentiment_model, vectorizer = train_sentiment_model(movie_reviews_df)
    test_review = "That movie was really bad, i don't prefer it anyone"
    print("Test Review Prediction:", predict_movie_review(sentiment_model, vectorizer, test_review))

    # Problem 5
    product_reviews_df = create_product_reviews_dataframe()
    print("Product Reviews DataFrame:")
    display(product_reviews_df)
    product_model = train_product_review_model(product_reviews_df)
    test_reviews = ["Valuable for price", "Good product"]
    transformed_reviews = product_model.named_steps["vectorizer"].transform(test_reviews)
    print("Transformed Test Reviews:\n", transformed_reviews)
