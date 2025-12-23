Data Files for PodSpectrum

This folder contains the core data files used by the PodSpectrum Streamlit app.

Included files

1. metadata.json
  Final podcast metadata used by the app. Contains series and episode information (titles, genres, release dates, durations, etc.).

2. podcast_episodes_with_topics_embeddings.pkl
  Pickle file with the processed dataset: SBERT embeddings for each episode plus associated topics and IDs. This is what the recommender loads to compute similarities.

3. podspectrum_feedback.csv 
  CSV file where the app stores user feedback (likes, dislikes, ratings, reviews) at runtime. It may be empty initially; the app will create/update it.

Notes

- Only the final data needed to run the app is included here.  
- Intermediate or experimental datasets used during development are not required and are therefore not part of this artefact.
