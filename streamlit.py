import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_data
def load_cleaned_data(path='cleaned_data.csv'):
    return pd.read_csv(path, index_col=0)

@st.cache_resource
def load_encoded_data_and_encoder(encoded_path='encoded_data.csv', encoder_path='encoder.pkl'):
    encoded_df = pd.read_csv(encoded_path, index_col=0)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    return encoded_df, encoder

def recommend_restaurants(user_input, encoder, encoded_df, cleaned_df, top_n=5):
    # Filter by city
    city_df = cleaned_df[cleaned_df['city'] == user_input['city']]

    # Filter by cuisine: keep rows where any user cuisine is in restaurant's cuisine list
    mask_cuisine = city_df['cuisine'].apply(
        lambda x: any(c.strip().lower() in [uc.lower() for uc in user_input['cuisines']] 
                      for c in x.split(','))
    )
    filtered_df = city_df[mask_cuisine]

    if filtered_df.empty:
        return pd.DataFrame()  # no matches for chosen filters

    filtered_indices = filtered_df.index
    filtered_encoded = encoded_df.loc[filtered_indices]

    # Build user feature vector
    df_user = pd.DataFrame({
        'name': [''],  # placeholder
        'city': [user_input['city']],
        'cuisine': [",".join(user_input['cuisines'])],
        'rating': [user_input['rating']],
        'rating_count': [filtered_encoded.iloc[:,1].mean() if not filtered_encoded.empty else 0],
        'cost': [user_input['cost']]
    })

    user_cat_encoded = encoder.transform(df_user[['city', 'cuisine']])
    user_num = df_user[['rating', 'rating_count', 'cost']].to_numpy()
    user_features = np.hstack([user_num, user_cat_encoded])

    similarities = cosine_similarity(user_features, filtered_encoded)[0]

    top_indices_local = similarities.argsort()[::-1][:top_n]
    recommended_indices = filtered_encoded.index[top_indices_local]

    return cleaned_df.loc[recommended_indices]

def main():
    st.title("Restaurant Recommendation System")

    cleaned_df = load_cleaned_data()
    encoded_df, encoder = load_encoded_data_and_encoder()

    city = st.selectbox("Select City:", sorted(cleaned_df['city'].unique()))
    all_cuisines = sorted({c.strip() for sublist in cleaned_df['cuisine'].str.split(',') for c in sublist})
    cuisines = st.multiselect("Select Preferred Cuisines:", options=all_cuisines)
    rating = st.slider("Minimum Rating:", 0.0, 5.0, 3.5)
    cost = st.slider("Maximum Cost ($):", int(cleaned_df['cost'].min()), int(cleaned_df['cost'].max()), int(cleaned_df['cost'].median()))

    if st.button("Get Recommendations"):
        if not cuisines:
            st.error("Please select at least one cuisine.")
        else:
            user_input = {
                'city': city,
                'cuisines': cuisines,
                'rating': rating,
                'cost': cost
            }
            results = recommend_restaurants(user_input, encoder, encoded_df, cleaned_df, top_n=10)
            if results.empty:
                st.warning("No restaurants found matching your criteria.")
            else:
                st.subheader(f"Top {len(results)} Restaurants in {city} matching your preferences:")
                st.dataframe(results[['name','city','cuisine','rating','cost','link']])

if __name__ == "__main__":
    main()
