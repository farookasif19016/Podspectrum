# PodSpectrum – Personality-Aware Podcast Recommendation System

🚀 **Live Demo**: Try the app now → [PodSpectrum on Streamlit](https://podspectrum-rojjt322kus7rcgguxd6ka.streamlit.app/)

This repository contains the code and core data for **PodSpectrum**, a personality-aware podcast recommendation system developed as part of an MSc dissertation.

The system recommends podcast episodes based on the listener's **OCEAN (Big Five) personality traits** combined with thematic content analysis using NLP (SBERT sentence embeddings, BERTopic topic modeling, cosine similarity, and trait-weighted scoring). It addresses the cold-start problem through content-based filtering and includes a user-friendly Streamlit interface.

## Key Features
- Personality profiling via OCEAN sliders (Curiosity/Openness, Attention/Conscientiousness, Sociability/Extraversion, Agreeableness, Moodiness/Neuroticism)
- Semantic + trait-hybrid recommendation engine
- Genre and topic filtering
- Feedback collection and playlist saving
- Explainable trait-semantic agreement metric

## Project Structure
- `app/test_2.py` — Main Streamlit application
- `data/` — Metadata, precomputed embeddings, feedback CSV
- `images/` — Logo and static images
- `notebooks/` — Jupyter notebooks used for embedding preparation
- `requirements.txt` — Python dependencies

## Setup & Running Locally
1. (Optional) Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. To run the app
   ```bash
   streamlit run app/test_2.py
   
## Screenshot
<img width="1918" height="872" alt="Output_Interface" src="https://github.com/user-attachments/assets/9935bc53-13fe-444b-b133-f3dfd60bafe5" />
                                                     

## Tech Stack

Python
Streamlit
Pandas, NumPy
scikit-learn
Sentence-Transformers (SBERT)
BERTopic

Open to feedback, collaborations, and junior data science opportunities! 🚀
