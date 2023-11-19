import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random

data = pd.read_csv('Food_Recipe.csv')

# make into data frame
df = pd.DataFrame(data)

# term frequency - inverse document frequency
tfidf_vectorizer = TfidfVectorizer()

# make all content into one
df['content'] = (
    df['name'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['cuisine'].fillna('') + ' ' +
    df['course'].fillna('') + ' ' +
    df['diet'].fillna('') + ' ' +
    df['ingredients_name'].fillna('') + ' ' +
    #df['ingredients_quantity'].fillna('') + ' ' +
    #df['prep_time (in mins)'].fillna('').astype(str) + ' ' +
    #df['cook_time (in mins)'].fillna('').astype(str) + ' ' +
    df['instructions'].fillna('')
    #df['image_url'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
)
print(df)
# fit & transform the content w tfidf vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

# find cosine similarity btwn all foods w each other
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recs(name, cosine_sim=cosine_sim):
    # get id of this name from df
    for idx, x in enumerate(df['name']):
        x = x.strip()
        x = x.lower()

        name = name.strip()
        name = name.lower()

        if x == name:
            final_idx = idx
            break

    # get sim scores of this name w id
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

print(get_recs("Healthy Yogurt Parfait with Oats and Fresh Fruits Recipe"))
