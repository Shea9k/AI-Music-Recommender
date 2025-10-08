# AI Music Recommender
A Python Streamlit application that recommends songs based on audio features from a Spotify dataset. Users select a song from a dropdown menu, and the app returns similar tracks using cosine similarity on features such as danceability, energy, and valence.

## Features
- Song recommendations based on audio feature similarity  
- Automatic handling of messy text and special characters  
- Normalized and scaled features for accurate similarity calculations  
- Simple dropdown selection for quick interaction  
- Optimized performance with a limited song subset
  
## Installation
1. **Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd MusicRec

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

## Usage
Run the Streamlit application:
   ```bash
streamlit run app.py
```
A browser window will open automatically. Select a song from the dropdown to view similar song recommendations.

## Dataset
The application uses `spotify_features.csv`, a public Spotify dataset containing audio features and metadata for popular songs.  
Columns include: `track_name`, `artist(s)_name`, `bpm`, `key`, `mode`, `danceability_%`, `energy_%`, `valence_%`, and others.  
Only a subset of songs is shown in the dropdown for performance optimization.

## How It Works
1. Loads and cleans the dataset (`spotify_features.csv`)
2. Fixes text encoding issues using `ftfy`
3. Scales and normalizes numerical audio features
4. Computes cosine similarity between songs
5. Returns the top N most similar songs for the selected track

## Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit
- ftfy

## Notes
- Cosine similarity is calculated on normalized audio features.
- Song and artist names are cleaned to remove encoding issues.
- The dropdown limits the number of songs to improve performance.
