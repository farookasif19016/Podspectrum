import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import base64
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="PodSpectrum Test_2", layout="wide")

# --- SIDEBAR BRANDING BAR (NO HAMBURGER) ---
# Relative path to the logo (from repo root)
logo_path = "images/PodSpectrum LOGO.png"

# Simpler and recommended way – Streamlit can display images directly
st.sidebar.image(logo_path, width=200)

# Alternative base64 method (kept in case you want custom styling)
# with open(logo_path, "rb") as imgfile:
#     img_bytes = imgfile.read()
# img_b64 = base64.b64encode(img_bytes).decode()
# img_src = f"data:image/png;base64,{img_b64}"
# st.sidebar.markdown(
#     f"""
#     <div style="display: flex; align-items: center; gap: 0.32em; margin: .1em 0 .5em 0; min-height:39px;">
#         <img src="{img_src}" alt="Pod Spectrum Logo"
#              style="height:36px; margin:-2px .19em 0 0; vertical-align:middle; display:inline-block;">
#         <span style="font-size:1.42em; color:#fff; font-family:sans-serif; font-weight:700; letter-spacing:.01em;vertical-align:middle;display:inline-block;">
#             Pod Spectrum
#         </span>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# Relative path to feedback CSV
FEEDBACK_CSV = "data/podspectrum_feedback.csv"

def traits_to_key(cur, att, soc, agr, moo):
    return f"{cur:.2f}_{att:.2f}_{soc:.2f}_{agr:.2f}_{moo:.2f}"

def save_feedback(traits_key, episode_id, like, dislike, not_int, rating, review):
    entry = {
        "traits_key": traits_key, "episode_id": episode_id,
        "like": like, "dislike": dislike, "not_interested": not_int,
        "rating": rating, "review_text": review,
        "timestamp": datetime.utcnow().isoformat()
    }
    for attempt in range(3):
        try:
            if os.path.exists(FEEDBACK_CSV):
                df = pd.read_csv(FEEDBACK_CSV)
            else:
                df = pd.DataFrame()
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            df.to_csv(FEEDBACK_CSV, index=False)
            return
        except PermissionError:
            if attempt < 2:
                import time
                time.sleep(0.5)
            else:
                st.error(f"⚠️ Could not save feedback. Please close any programs using the file.")
                return
        except Exception as e:
            st.error(f"⚠️ Error saving feedback: {str(e)}")
            return

def get_feedback(traits_key, episode_id):
    if not os.path.exists(FEEDBACK_CSV):
        return pd.DataFrame()
    df = pd.read_csv(FEEDBACK_CSV)
    return df[(df['traits_key'] == traits_key) & (df['episode_id'] == episode_id)]

# Session state initialization (unchanged)
if "onboard_step" not in st.session_state:
    st.session_state["onboard_step"] = 0
if "tour_dismissed" not in st.session_state:
    st.session_state["tour_dismissed"] = False

for key, val in [
    ("playlist", []), ("user_feedback", {}), ("rec_ready", False), ("recommendations", []), ("podcast_details", {}),
    ("recommendation_counts", {}), ("series_count", {}), ("already_seen_series", set()), ("already_seen_episodes", set()),
    ("last_curiosity", None), ("last_attention", None), ("last_sociability", None),
    ("last_agreeableness", None), ("last_moodiness", None), ("last_selected_genres", None),
    ("last_topics_selected", None), ("last_num_recs", None)
]:
    if key not in st.session_state:
        st.session_state[key] = val

st.sidebar.header("Personalization Traits & Preferences")
curiosity = st.sidebar.slider("Curiosity", 0.0, 1.0, 0.6, 0.01)
attention = st.sidebar.slider("Attention to Detail", 0.0, 1.0, 0.5, 0.01)
sociability = st.sidebar.slider("Sociability", 0.0, 1.0, 0.5, 0.01)
agreeableness = st.sidebar.slider("Agreeableness", 0.0, 1.0, 0.5, 0.01)
moodiness = st.sidebar.slider("Moodiness", 0.0, 1.0, 0.3, 0.01)

traits_key = traits_to_key(curiosity, attention, sociability, agreeableness, moodiness)

# Fixed: Relative path to pickle file
df = pd.read_pickle("data/podcast_episodes_with_topics_embeddings.pkl")

# Fixed: Relative path to metadata JSON
with open("data/metadata.json", "r", encoding="utf-8") as metafile:
    podcast_details_all = json.load(metafile)

topic_to_genre = {
    "crime": "Crime", "unsolved": "Crime", "case": "Crime", "justice": "Crime", "serial killer": "Crime",
    "history": "History", "anatomy": "Science", "science": "Science", "biology": "Science",
    "medicine": "Science", "evolution": "Science", "astro": "Science", "mars": "Science",
    "exploration": "Adventure", "expedition": "Adventure", "survival": "Adventure",
    "mountain": "Adventure", "arctic": "Adventure", "antarctica": "Adventure",
    "relationship": "Society", "family": "Society", "wellbeing": "Society", "support": "Society",
    "comedy": "Comedy", "banter": "Comedy", "improv": "Comedy",
    "hollywood": "Entertainment", "celebrity": "Entertainment", "music": "Entertainment"
}
def map_topics_to_genres(topics):
    genres = set()
    for topic in topics if isinstance(topics, list) else [str(topics)]:
        t = topic.lower()
        for k in topic_to_genre:
            if k in t:
                genres.add(topic_to_genre[k])
    return list(genres) if genres else ["Other"]

df['genres'] = df['topics'].apply(map_topics_to_genres)
all_genres = sorted({g for genres in df['genres'] for g in genres if pd.notnull(g)})
all_available_topics = sorted({str(t) for ts in df['topics'] for t in (ts if isinstance(ts, list) else [ts])})

selected_genres = st.sidebar.multiselect("Genres", all_genres, default=[])
topics_selected = st.sidebar.multiselect("Explore Topics", all_available_topics, default=[])
num_recs = st.sidebar.slider("How Many Recommendations?", min_value=5, max_value=len(df), value=15, step=5)

SPOTLIGHT_BANNERS = [
    ("**Step 1: Personality Sliders (OCEAN model explained)**\n\n"
     "- **Curiosity / Openness**: Like variety, imagination, new ideas?\n"
     "- **Attention to Detail / Conscientiousness**: Prefer structure, thoroughness, order?\n"
     "- **Sociability / Extraversion**: Enjoy lively, social content?\n"
     "- **Agreeableness**: Like kindness, friendly or positive tone?\n"
     "- **Moodiness / Neuroticism**: Prefer deep, intense, or emotional topics?\n", "sliders"),
    ("**Step 2: Genres**\n\nPick genres, or leave blank for all.", "genres"),
    ("**Step 3: Topics**\n\nOptionally pick topics, or SKIP for all.", "topics"),
    ("**Step 4: Amount**\n\nHow many recommendations do you want?", "num"),
    ("**Step 5: Click 'Get My Recommendations'** to begin.", "run"),
]
def show_banner(idx):
    msg, key = SPOTLIGHT_BANNERS[idx]
    with st.container():
        st.info(msg)
        cols = st.columns([1, 1, 1])
        prev_disabled = idx == 0
        next_disabled = idx >= len(SPOTLIGHT_BANNERS) - 1
        if cols[0].button("Prev", key=f"info_prev_{idx}", disabled=prev_disabled):
            st.session_state["onboard_step"] = max(st.session_state["onboard_step"]-1, 0)
            st.rerun()
        if cols[1].button("Next", key=f"info_next_{idx}", disabled=next_disabled):
            st.session_state["onboard_step"] = min(st.session_state["onboard_step"]+1, len(SPOTLIGHT_BANNERS)-1)
            st.rerun()
        if cols[2].button("Skip Tutorial", key=f"info_skip_{idx}"):
            st.session_state["tour_dismissed"] = True
            st.rerun()

if not st.session_state["tour_dismissed"]:
    show_banner(st.session_state["onboard_step"])

if st.button("Replay Tutorial", key="replay_tour"):
    st.session_state["tour_dismissed"] = False
    st.session_state["onboard_step"] = 0
    st.rerun()

traits_changed = (
    st.session_state["last_curiosity"] != curiosity or
    st.session_state["last_attention"] != attention or
    st.session_state["last_sociability"] != sociability or
    st.session_state["last_agreeableness"] != agreeableness or
    st.session_state["last_moodiness"] != moodiness or
    st.session_state["last_selected_genres"] != selected_genres or
    st.session_state["last_topics_selected"] != topics_selected or
    st.session_state["last_num_recs"] != num_recs
)
if traits_changed:
    st.session_state["rec_ready"] = False
    st.session_state["recommendations"] = []
    st.session_state["last_curiosity"] = curiosity
    st.session_state["last_attention"] = attention
    st.session_state["last_sociability"] = sociability
    st.session_state["last_agreeableness"] = agreeableness
    st.session_state["last_moodiness"] = moodiness
    st.session_state["last_selected_genres"] = selected_genres
    st.session_state["last_topics_selected"] = topics_selected
    st.session_state["last_num_recs"] = num_recs

def trait_influence(row, curiosity, attention, sociability, agreeableness, moodiness):
    topic_strs = [str(t).lower() for t in row.get('topics',[])] if isinstance(row.get('topics',[]), list) else [str(row.get('topics','')).lower()]
    op_keywords = ["exploration", "ideas", "philosophy", "curiosity", "innovation", "abstract", "discovery"]
    con_keywords = ["method", "planning", "tutorial", "structure", "education", "instruction", "detail"]
    ext_keywords = ["interview", "banter", "roundtable", "improv", "panel", "audience", "loud"]
    agr_keywords = ["family", "empathy", "relationship", "wellbeing", "support", "teamwork"]
    neu_keywords = ["crime", "drama", "intensity", "mystery", "survival", "anxiety", "stress"]
    op_neg = ["routine", "repetition", "predictable"]
    con_neg = ["improv", "unscripted", "chaos"]
    ext_neg = ["solitude", "reflection", "solo", "quiet"]
    agr_neg = ["debate", "conflict", "betrayal", "competition"]
    neu_neg = ["support", "comfort", "positivity", "calm"]
    op_val = (sum([kw in topic_strs for kw in op_keywords]) - sum([kw in topic_strs for kw in op_neg])) * curiosity
    con_val = (sum([kw in topic_strs for kw in con_keywords]) - sum([kw in topic_strs for kw in con_neg])) * attention
    ext_val = (sum([kw in topic_strs for kw in ext_keywords]) - sum([kw in topic_strs for kw in ext_neg])) * sociability
    agr_val = (sum([kw in topic_strs for kw in agr_keywords]) - sum([kw in topic_strs for kw in agr_neg])) * agreeableness
    neu_val = (sum([kw in topic_strs for kw in neu_keywords]) - sum([kw in topic_strs for kw in neu_neg])) * moodiness
    blend = 4 * (op_val**2) + 4 * (con_val**2) + 4 * (ext_val**2) + 4 * (agr_val**2) + 4 * (neu_val**2)
    return blend

def get_user_embedding(selected_topics, df):
    if selected_topics:
        topic_mask = df['topics'].apply(lambda ts: any(t in str(ts).lower() for t in selected_topics))
        if topic_mask.sum() > 0:
            return np.mean(np.vstack(df.loc[topic_mask, 'embedding'].to_numpy()), axis=0)
    return np.mean(np.vstack(df['embedding'].to_numpy()), axis=0)

def recommend_semantic_diverse(
    df, user_embedding, curiosity, attention, sociability,
    agreeableness, moodiness, selected_topics=None, top_n=10, diversity_penalty=0.1
):
    unseen_df = df[
        ~df['series'].isin(st.session_state["already_seen_series"]) &
        ~df['episode_id'].isin(st.session_state["already_seen_episodes"])
    ]
    rec_pool = unseen_df if len(unseen_df) >= top_n else df

    episode_embeddings = np.vstack(rec_pool['embedding'].to_numpy())
    similarities = cosine_similarity(episode_embeddings, user_embedding.reshape(1, -1)).flatten()

    df_copy = rec_pool.copy()
    df_copy['cog_trait_score'] = df_copy.apply(
        lambda row: trait_influence(row, curiosity, attention, sociability, agreeableness, moodiness),
        axis=1
    )

    sim_scaler = MinMaxScaler()
    trait_scaler = MinMaxScaler()
    scaled_similarity = sim_scaler.fit_transform(similarities.reshape(-1, 1)).flatten()
    scaled_trait = trait_scaler.fit_transform(df_copy['cog_trait_score'].to_numpy().reshape(-1, 1)).flatten()

    # normalized components
    df_copy["sem_score"] = scaled_similarity
    df_copy["trait_score_scaled"] = scaled_trait

    # hybrid score (traits dominate more)
    final_score = 0.85 * scaled_trait + 0.15 * scaled_similarity
    df_copy['similarity'] = final_score + np.random.rand(len(df_copy)) * 1e-6

    # agreement metric with mean-based thresholds
    trait_thr = df_copy["trait_score_scaled"].mean()
    sem_thr   = df_copy["sem_score"].mean()
    df_copy["trait_label"] = (df_copy["trait_score_scaled"] >= trait_thr).astype(int)
    df_copy["sem_label"]   = (df_copy["sem_score"]         >= sem_thr).astype(int)
    df_copy["agree"] = (df_copy["trait_label"] == df_copy["sem_label"]).astype(int)

    # select recommendations
    recs = df_copy.nlargest(top_n, "similarity")

    # agreement over top-K only
    top_agreement = float(recs["agree"].mean())
    st.session_state["agreement_accuracy"] = top_agreement

    for _, row in recs.iterrows():
        eid = row.get('episode_id', 'N/A')
        sid = row.get('series', 'N/A')
        st.session_state["already_seen_episodes"].add(eid)
        st.session_state["already_seen_series"].add(sid)

    return [dict(row) for _, row in recs.iterrows()]

def render_episode_card(episode, series_metadata, card_number):
    st.write(f"### Recommendation #{card_number}")
    series_cols = st.columns([1, 4])
    with series_cols[0]:
        img_displayed = False
        img_path = series_metadata.get("thumbnail", "")
        if isinstance(img_path, str) and img_path.strip() != "":
            if os.path.exists(img_path):
                st.image(img_path, width=100)
                img_displayed = True
            elif img_path.lower().startswith("http"):
                st.image(img_path, width=100)
                img_displayed = True
        if not img_displayed:
            st.image("https://static.thenounproject.com/png/104062-200.png", width=100)
    with series_cols[1]:
        st.write("**SERIES**")
        st.write(f"#### {series_metadata.get('title', 'N/A')}")
        st.write(f"**Genres:** {series_metadata.get('genres', 'N/A')}")
        st.write(f"**Duration:** {series_metadata.get('duration', 'N/A')}")
        st.write(f"**Release:** {series_metadata.get('release_date', 'N/A')}")
        st.write(f"**Producer/Publisher:** {series_metadata.get('creators', 'N/A')}")
        st.write(f"**Hosts/Authors:** {series_metadata.get('hosts', 'N/A')}")
        st.write(f"**Impact/Rating:** {series_metadata.get('rating', 'N/A')}")
    st.divider()
    st.write("**RECOMMENDED EPISODE**")
    st.write(f"#### Episode {episode.get('episode_id', 'N/A')}: {episode.get('title', 'N/A')}")
    st.write("**Episode Summary:**")
    st.write(f"{episode.get('summary', 'N/A')}")
    st.write(f"**Topics Covered:** {', '.join(episode.get('topics', [])) if isinstance(episode.get('topics', []), list) else episode.get('topics', '')}")
    st.write(f"**Episode Release Date:** {episode.get('release_date', 'N/A')} | **Episode Duration:** {episode.get('length', 'N/A')}")
    if episode.get('transcript') and str(episode.get('transcript', '')).strip():
        with st.expander("View Episode Transcript"):
            transcript_disp = episode.get('transcript', '')
            st.text(transcript_disp[:2000] + "..." if len(transcript_disp) > 2000 else transcript_disp)
    st.divider()
    podcast_link = series_metadata.get('url', '#')
    ref_link = series_metadata.get('more_info_url', '#')
    st.write(f"**More Info:** [Listen on Podcast Platform]({podcast_link}) | [Reference]({ref_link})")
    with st.expander("💬 Leave a Review & See What Others with Your Settings Think", expanded=False):
        st.markdown("#### Leave a Detailed Review")
        with st.form(f"feedback_form_{episode['episode_id']}"):
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            with col1:
                fb_like = st.checkbox("Like", key=f"fb_like_{episode['episode_id']}")
            with col2:
                fb_dislike = st.checkbox("Dislike", key=f"fb_dislike_{episode['episode_id']}")
            with col3:
                fb_notint = st.checkbox("Not Interested", key=f"fb_notint_{episode['episode_id']}")
            with col4:
                in_playlist = any(e['episode_id'] == episode['episode_id'] for e in st.session_state["playlist"])
                pl_btn_label = "Add to Playlist" if not in_playlist else "Remove from Playlist"
                if st.form_submit_button(pl_btn_label):
                    if not in_playlist:
                        st.session_state["playlist"].append({"episode_id": episode['episode_id'], "title": episode["title"], "series": episode.get("series","")})
                        st.success("Added to your playlist.")
                    else:
                        st.session_state["playlist"] = [e for e in st.session_state["playlist"] if e['episode_id'] != episode['episode_id']]
                        st.success("Removed from your playlist.")
                    st.rerun()
            fb_rating = st.slider("Rate (stars)", 1, 5, 3, key=f"fb_rate_{episode['episode_id']}")
            fb_review = st.text_area("Write your review", key=f"fb_review_{episode['episode_id']}", height=110)
            fb_submit = st.form_submit_button("Submit Detailed Feedback")
            if fb_submit:
                save_feedback(traits_key, episode['episode_id'], fb_like, fb_dislike, fb_notint, fb_rating, fb_review)
                st.success("Your detailed feedback has been recorded!")
                st.rerun()
        st.markdown("---")
        st.markdown("#### 💭 What Others with Your Settings Think:")
        feedback_df = get_feedback(traits_key, episode['episode_id'])
        if feedback_df.empty:
            st.info("No feedback yet from others with your trait combination.")
        else:
            for idx, row in feedback_df.iterrows():
                with st.container():
                    col_rating, col_rest = st.columns([1, 4])
                    with col_rating:
                        st.write(f"⭐ {row['rating']}/5")
                    with col_rest:
                        if row['review_text'] and str(row['review_text']).strip():
                            st.write(f"_{row['review_text']}_")
                    st.caption(f"👍 {row['like']} | 👎 {row['dislike']} | ❌ {row['not_interested']} | 📅 {row['timestamp'][:10]}")
                st.divider()

with st.sidebar.expander("My Playlist", expanded=True):
    playlist = st.session_state.get("playlist", [])
    if playlist:
        for ep in playlist:
            st.markdown(f"- **{ep['title']}** <span style='font-size:smaller'>({ep.get('series','')})</span>", unsafe_allow_html=True)
            if st.button("Remove", key=f"sidebar_rem_{ep['episode_id']}"):
                st.session_state["playlist"] = [e for e in playlist if e['episode_id'] != ep['episode_id']]
                st.rerun()
    else:
        st.write("Your saved episodes will appear here.")

if st.button("Get My Recommendations"):
    st.session_state["recommendation_counts"] = {}
    st.session_state["series_count"] = {}
    st.session_state["recommendations"] = []
    st.session_state["rec_ready"] = False
    working_df = df.copy()
    if selected_genres:
        working_df = working_df[working_df['genres'].apply(lambda gs: any(g in selected_genres for g in (gs if isinstance(gs, list) else [gs])))]
    user_vec = get_user_embedding(topics_selected, working_df)
    recs = recommend_semantic_diverse(
        working_df, user_vec, curiosity, attention, sociability,
        agreeableness, moodiness, topics_selected, top_n=num_recs, diversity_penalty=0.3
    )
    st.session_state["recommendations"] = recs
    st.session_state["rec_ready"] = True
    st.session_state["podcast_details"] = podcast_details_all

if st.session_state.get("rec_ready", False):
    recs = st.session_state.get("recommendations", [])
    podcast_details = st.session_state.get("podcast_details", {})

    # show agreement metric if it was computed
    if "agreement_accuracy" in st.session_state:
        acc = st.session_state["agreement_accuracy"]
        st.metric(
            "Trait–semantic agreement",
            f"{acc * 100:.1f}%",
            help="Share of candidate episodes where personality-based and semantic scores both classify the episode as high or low."
        )

    if recs:
        for idx, rec in enumerate(recs, 1):
            series_meta = podcast_details.get(rec.get('series','N/A'))
            if series_meta:
                render_episode_card(rec, series_meta, idx)
            else:
                st.warning(
                    f"Series '{rec.get('series','N/A')}' metadata not found. "
                    f"Skipping {rec.get('title','N/A')}."
                )
    else:
        st.warning("No recommendations found with your current settings. Try changing traits or genres!")

