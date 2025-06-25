import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv("Medicine_Details.csv")
df.dropna(subset=['Composition', 'Uses', 'Side_effects'], inplace=True)
df['Text'] = df['Composition'] + ' ' + df['Uses'] + ' ' + df['Side_effects']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Text'])

knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(X)

def recommend(medicine_name):
    try:
        index = df[df['Medicine Name'].str.lower() == medicine_name.lower()].index[0]
        distances, indices = knn.kneighbors(X[index])
        recommended = df.iloc[indices[0][1:]]
        return recommended[['Medicine Name', 'Composition', 'Uses']]
    except:
        return pd.DataFrame([{'Medicine Name': 'Not found', 'Composition': '', 'Uses': ''}])


st.title("💊 Medicine Recommendation System")


selected_medicine = st.selectbox(
    "Choose a medicine:",
    options=df['Medicine Name'].unique()
)

if st.button("Recommend"):
    results = recommend(selected_medicine)
    st.write("### Recommended Alternatives:")
    st.dataframe(results)
