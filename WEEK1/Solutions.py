#%% WEEK1 Solutions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% PROBLEM 1 SOLUTION
#
##
###
array=np.random.randint(1,50,size=(5,4))
print(array)

# %%
# extracting anti-diagonal elements
anti_diag = np.array([array[i, array.shape[1]-i-1] for i in range(array.shape[1])])
print(anti_diag)
# %%
print(np.max(array, axis = 1)) # max elements along the row
# %%
print(np.mean(array)) # mean of all elements
# %%
new_arr = array[array <= np.mean(array)] # new array with all the elements less than or equal to mean
print(new_arr)
# %%
def numpy_boundary_transversal(matrix):
    shape = matrix.shape
    lst1 = matrix[0, :].tolist()
    lst2 = (matrix[shape[0]-1, :].tolist())[::-1]
    lst3 = matrix[1:shape[0]-1, shape[1]-1].tolist()
    lst4 = (matrix[1:shape[0]-1, 0].tolist())[::-1]
    return lst1 + lst3 + lst2 + lst4
    
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix)
print(numpy_boundary_transversal(matrix))
#%% PROBLEM 2 SOLUTION
#
##
###
arr2 = np.random.uniform(0, 10, size = 20)
print(arr2)
# %%
rounded = np.round(arr2, decimals = 2)
print(rounded)
# %%
print(f"Maximum of array is: {np.max(rounded)} ")
print(f"Minimum of array is: {np.min(rounded)} ")
print(f"Median of array is: {np.median(rounded)} ")
# %%
# replacing all elements less than 5 by their squares
less_than = np.where(rounded < 5 , rounded**2, rounded)
print(less_than)
# %%
# function that take an 1d array as input and return an alternate sorted array as 1st element as shortest, 2nd is largest, 3rd is 2nd smallest, 4th is 2nd largest and so on
def numpy_alternate_sort(array):
    array.sort()
    opt = []
    i = 0
    j = array.shape[0]-1
    while(i <j):
        opt.append(array[i])
        opt.append(array[j])
        i +=1
        j -=1
    if (array.shape[0])%2 ==0:
        return np.array(opt)
    else:
        opt.append(array[i])
        return np.array(opt)
        
array = np.array([1,6,3,9,10,12,5,4])  
print(numpy_alternate_sort(array))
#%% PROBLEM 3 SOLUTION
#
##
###
df = pd.DataFrame({"Name": ["S1", "S2", "S3", "S4", "S5", "S6","S7", "S8","S9","S10"],
                  "Subject" : ["Math", "CS", "Physics", "Chem", "Math", "CS", "Chem", "CS", "Physics", "Math"],
                  "Score" : np.random.randint(50, 100, size = 10).tolist(),
                  "Grade" : np.array([""]*10, dtype = object)})
display(df)
# %%
# Define conditions and corresponding grades
conditions = [
    (df['Score'] >= 90),
    (df['Score'] >= 80) & (df['Score'] < 90),
    (df['Score'] >= 70) & (df['Score'] < 80),
    (df['Score'] >= 60) & (df['Score'] < 70),
    (df['Score'] < 60)
]

grades = ['A', 'B', 'C', 'D', 'F']

# Assign grades
df['Grade'] = np.select(conditions, grades, default='F') 

# Display result
display(df)
# %%
df = df.sort_values(by = "Score", ascending = False, ignore_index = True)
display(df)
# %%
subject_avg_df = df.groupby("Subject", as_index=False)["Score"].mean()
display(subject_avg_df)
# %%
print(np.array(df["Score"]))
# %%
def pandas_filter_pass(dataframe):
    df2 = dataframe[(dataframe["Grade"] == "A") | (dataframe["Grade"] == "B")]
    return df2
df2 = pandas_filter_pass(df)
print(df2)

#%% PROBLEM 4 SOLUTION
#
##
###
import pandas as pd
import random

# Sample templates
positive_templates = [
    "Absolutely loved it!",
    "A fantastic experience from start to finish.",
    "Great acting and brilliant storyline.",
    "One of the best movies I've seen this year.",
    "Incredible direction and cinematography.",
    "A masterpiece. Would highly recommend.",
    "Emotionally moving and beautifully shot.",
    "Engaging, entertaining, and very well made.",
    "Excellent pacing and character development.",
    "An unforgettable cinematic journey.",
    "Loved every minute of it.",
    "The plot was thrilling and unpredictable.",
    "The cast did a phenomenal job.",
    "A perfect blend of drama and action.",
    "Hilarious and heartfelt at the same time.",
    "Simply amazing. I'd watch it again!",
    "Beautifully executed.",
    "A must-watch for everyone.",
    "Outstanding performance by the lead actor.",
    "This movie exceeded my expectations.",
    "A solid 10 out of 10!",
    "Top-notch production value.",
    "A delight to watch.",
    "The emotional scenes hit hard.",
    "Visually stunning and thought-provoking.",
]

negative_templates = [
    "Terribly disappointing.",
    "A complete waste of time.",
    "Poor acting and weak storyline.",
    "I expected much more from this movie.",
    "Failed to impress in any aspect.",
    "The plot was slow and boring.",
    "Couldn’t finish it. It was that bad.",
    "Flat characters and meaningless dialogue.",
    "Not worth the hype.",
    "Disjointed scenes and lazy writing.",
    "Predictable and dull.",
    "Felt like a bad TV episode.",
    "Unconvincing performances.",
    "The script was all over the place.",
    "I kept waiting for it to get better, it didn’t.",
    "Nothing memorable at all.",
    "Sloppy editing ruined it.",
    "Forgettable and bland.",
    "Regret watching this.",
    "It dragged on forever.",
    "Bad CGI and worse acting.",
    "Totally underwhelming.",
    "No chemistry between the leads.",
    "Tried to be deep, but failed.",
    "One of the worst movies I’ve seen.",
]

# Duplicate and randomize to make 50 each
positive_reviews = random.choices(positive_templates, k=50)
negative_reviews = random.choices(negative_templates, k=50)

# Combine and label
reviews = positive_reviews + negative_reviews
sentiments = ['positive'] * 50 + ['negative'] * 50

# Shuffle the dataset
combined = list(zip(reviews, sentiments))
random.shuffle(combined)

# Create DataFrame
df = pd.DataFrame(combined, columns=['Review', 'Sentiment'])

# Display a few samples
display(df)


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

X = df["Review"]
y = df["Sentiment"]
vector = CountVectorizer(max_features = 500, stop_words = 'english') # as lowercase is already true no need to mention it
X = vector.fit_transform(X)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)

model = Pipeline([
    ("Model", MultinomialNB())
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy: {accuracy*100}%")

def predict_review_sentiment(trained_model, fitted_vectorizer, test_string):
    test = fitted_vectorizer.transform([test_string])
    y_pred = trained_model.predict(test)
    return y_pred[0]

s = "That movie was really bad, i don't prefer it anyone"
y_pred = predict_review_sentiment(model, vector, s)
print(y_pred)

#%% PROBLEM 5 SOLUTION
#
##
###
import pandas as pd
import random

# Example positive and negative feedback phrases
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
    "Missing parts", "Doesn't fit properly", "Feels flimsy", "Very slow", "Buttons don’t work",
    "Battery drains fast", "Not intuitive", "Disappointed with the quality", "Looks cheap", "Very annoying",
    "Uncomfortable", "Not user-friendly", "Regret buying it", "Barely works", "Returned it immediately",
    "Overheats quickly", "Feels like a toy", "No value for money", "Subpar build", "Bad experience"
]

# Randomly shuffle the lists
random.shuffle(good_phrases)
random.shuffle(bad_phrases)

# Create DataFrame
reviews = good_phrases[:50] + bad_phrases[:50]
labels = ['good'] * 50 + ['bad'] * 50

df = pd.DataFrame({'Review': reviews, 'Sentiment': labels})

# Optional: shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# Display dataframe
display(df)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

X = df["Review"]
y = df["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 1)
model = Pipeline([
    ("vector", TfidfVectorizer(max_features = 300, stop_words = 'english')),
    ("Linear", LogisticRegression(max_iter = 200))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(pred)

recall = recall_score(y_test, pred, pos_label='good')
precision = precision_score( y_test, pred, pos_label='good')
f1 = f1_score(y_test, pred, pos_label='good')

print(f"Recall Score: {recall}")
print(f"Precision Score: {precision}")
print(f"F1 Score: {f1}")

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

def text_preprocess_vectorize(list_texts, fitted_vectorizer):
    return fitted_vectorizer.transform(list_texts)

reviews = ["Valuable for price", "Good product"]
fit = text_preprocess_vectorize(reviews, vector)
print(fit)
# %%
