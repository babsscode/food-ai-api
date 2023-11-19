import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import os

def calculate_and_save_cosine_similarity(df, filename='cosine_similarity.npz'):
    tfidf_vectorizer = TfidfVectorizer()

    # make all content into one
    df['content'] = (
            df['name'].fillna('') + ' ' +
            df['description'].fillna('') + ' ' +
            df['cuisine'].fillna('') + ' ' +
            df['course'].fillna('') + ' ' +
            df['diet'].fillna('') + ' ' +
            df['ingredients_name'].fillna('') + ' ' +
            df['instructions'].fillna('')
    )

    # fit & transform the content w tfidf vectorizer
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

    # find cosine similarity between all foods with each other
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Save the cosine similarity matrix to a file
    np.savez_compressed(filename, cosine_sim=cosine_sim)


# Check if cosine similarity matrix file exists
cosine_similarity_file = 'cosine_similarity.npz'
# make data into data frame
data = pd.read_csv('Food_Recipe.csv')
df = pd.DataFrame(data)

if os.path.isfile(cosine_similarity_file):
    # If the file exists, load the cosine similarity matrix from the file
    with np.load(cosine_similarity_file) as data:
        cosine_sim = data['cosine_sim']
else:
    # If the file doesn't exist, perform the calculations
    calculate_and_save_cosine_similarity(df, cosine_similarity_file)
    # Load the cosine similarity matrix from the file
    with np.load(cosine_similarity_file) as data:
        cosine_sim = data['cosine_sim']

def get_recs(name, cosine_sim=cosine_sim, df=df):
    # get id of this name from df in a case-insensitive and whitespace-insensitive way
    idx = df.index[df['name'].str.strip().str.lower() == name.strip().lower()].tolist()

    if not idx:
        return []

    idx = idx[0]

    # get sim scores of this name with id
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort in highest to lowest
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # top 10 similar not excluding first (which is itself)
    sim_scores = sim_scores[1:11]

    # get index for each of the foods in top 10 sim_scores
    food_indices = [i[0] for i in sim_scores]

    # get name corresponding to index from df
    final_list = []
    for thename in df['name'].iloc[food_indices]:
        final_list.append(thename.split(' Recipe')[0])
    return final_list


# Test with different inputs
print(get_recs("Healthy Yogurt Parfait with Oats and Fresh Fruits Recipe"))
