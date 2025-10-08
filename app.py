"""
Music Recommendation Web App

Streamlit application that recommends songs based on audio features using cosine similarity.
Dataset: Spotify Features (public dataset).
"""

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import ftfy


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spotify_features.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("spotify_features.csv", encoding="latin-1")

    # Automatically fix messed-up characters in all text columns
    text_cols = ["track_name", "artist(s)_name"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ftfy.fix_text(str(x)))

    if df["key"].dtype == "object":
        key_map = {
            "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
            "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
        }
        df["key"] = df["key"].map(key_map)

    if df["mode"].dtype == "object":
        mode_map = {"Major": 1, "Minor": 0}
        df["mode"] = df["mode"].map(mode_map)

    feature_cols = [
        "bpm",
        "key",
        "mode",
        "danceability_%",
        "valence_%",
        "energy_%",
        "acousticness_%",
        "instrumentalness_%",
        "liveness_%",
        "speechiness_%",
    ]
    df = df.dropna(subset=feature_cols)

    return df


def recommend(song_name: str, df: pd.DataFrame, feature_cols: list, n: int = 5) -> pd.DataFrame:
    if song_name not in df["track_name"].values:
        return pd.DataFrame([{"track_name": "Not found", "artist(s)_name": "-"}])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    idx = df[df["track_name"] == song_name].index[0]
    song_features = X_scaled[idx].reshape(1, -1)
    sim_scores = cosine_similarity(song_features, X_scaled)[0]

    top_indices = sim_scores.argsort()[::-1][1 : n + 1]
    return df.iloc[top_indices][["track_name", "artist(s)_name"]]


def main():
    st.title("🎶 AI Music Recommender")
    st.write("Find songs similar to your favorite track based on audio features.")

    df = load_data()
    feature_cols = [
        "bpm",
        "key",
        "mode",
        "danceability_%",
        "valence_%",
        "energy_%",
        "acousticness_%",
        "instrumentalness_%",
        "liveness_%",
        "speechiness_%",
    ]

    sample_songs = df["track_name"].unique()[: len(df) // 2]
    song_choice = st.selectbox("Select a song:", sample_songs)

    if st.button("Recommend"):
        st.subheader(f"Songs similar to: {song_choice}")
        recs = recommend(song_choice, df, feature_cols)
        for _, row in recs.iterrows():
            st.write(f"🎵 {row['track_name']} — {row['artist(s)_name']}")


if __name__ == "__main__":
    main()










