import streamlit as st
import numpy as np
import json
from io import BytesIO
import base64

from sounds_like_utils import find_similar_songs, find_song_with_fuzzy_matching
from ner_pipeline import ner_pipeline
from data_utils import load_data
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from spotipy_util import init_spotify, get_spotify_track
from hf_utils import load_csv_from_hf, load_json_from_hf, load_binary_from_hf

@st.cache_resource
def load_model_and_data():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load CSVs from Hugging Face dataset
    df_scaled_features = load_csv_from_hf("scaled_data.csv", index_col=0)
    df_song_info = load_csv_from_hf("song_data.csv", index_col=0)

    # Load numpy arrays from Hugging Face dataset (binary)
    # ``load_binary_from_hf`` already returns a ``BytesIO`` object. Passing that
    # directly to ``np.load`` avoids wrapping it again (which would raise a
    # ``TypeError``) and keeps the data pointer intact.  Explicitly disallow
    # pickled data for security.
    song_embed_bytes = load_binary_from_hf("song_embeddings.npy")

    # Ensure np.load gets a file-like buffer
    song_embed = np.load(BytesIO(song_embed_bytes), allow_pickle=False)
    song_embeddings = normalize(song_embed)

    scaled_emotion_bytes = load_binary_from_hf("emotion_vectors.npy")
    scaled_emotion_means = np.load(BytesIO(scaled_emotion_bytes), allow_pickle=False)

    emotion_labels = load_json_from_hf("emotion_labels.json")

    return (
        embedder,
        df_scaled_features,
        df_song_info,
        song_embeddings,
        scaled_emotion_means,
        emotion_labels,
    )

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# App Setup
spotify = init_spotify()
st.set_page_config(page_title="SoundsLike", layout="wide")
st.title("🎵 SoundsLike: Music Recommendation Engine")
st.caption("Generate music recommendations from natural language prompts like *'sad songs like Moon by Kanye West'*")

# Load everything
embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotion_means, emotion_labels = load_model_and_data()

# Prompt Input
with st.container():
    st.subheader("💬 Enter Your Prompt")
    user_prompt = st.text_input("What vibe are you going for?", placeholder="e.g. sad songs like Moon by Kanye West")
    num_recs = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    print(f"Test 1: User Prompt: {user_prompt}")

    if st.button("🔍 Find Songs") and user_prompt.strip():
        # Attempt Fuzzy Matching
        print(f"Test 2: User Prompt: {user_prompt}")
        (result_tuple, closest_match) = find_song_with_fuzzy_matching(user_prompt, df_song_info, ner_pipeline)
        exact_match = result_tuple
        prompt_for_engine = user_prompt
        print(f"Test 3: User Prompt: {user_prompt}")

        if exact_match is not None and closest_match is True:
            print(f"[DEBUG] exact_match type: {type(exact_match)}")
            print(f"[DEBUG] exact_match contents:\n{exact_match}")
            matched_title = exact_match['Song']
            st.success(f"Found a direct match: {matched_title}. finding similar songs...")
            prompt_for_engine = matched_title
        else:
            st.info("No exact title found. searching by vibe...")

        print(f"Test 4: User Prompt: {user_prompt}")
        print(f"Test 5: User Prompt/Prompt for engine: {prompt_for_engine}")

        result = find_similar_songs(
            user_prompt=user_prompt,
            input_song=exact_match,
            num_recommendations=num_recs,
            ner_pipeline=ner_pipeline,
            embedder=embedder,
            df_scaled_features=df_scaled_features,
            df_song_info=df_song_info,
            song_embeddings=song_embeddings,
            scaled_emotion_means=scaled_emotion_means,
            emotion_labels=emotion_labels
        )

        if result:
            main_song = result["main_song"]
            recs = result["similar_songs"]
            recs = [main_song] + recs

            # Detected Entities
            st.markdown("### 🧠 Detected Entities")
            st.markdown(f"- {result['song_match_info']}")
            st.markdown(f"- {result['artist_match_info']}")
            st.markdown(f"- {result['mood_match_info']}")

            # Recommended Songs
            st.markdown("---")
            st.markdown("### 🎶 Recommended Songs")

            # (fix) only iterate when result exists
            for rec in recs:
                track = get_spotify_track(spotify, rec['title'], rec['artist'])

                if track:
                    album_img = track["album"]["images"][0]["url"]
                    external_url = track["external_urls"]["spotify"]
                    track_name = track["name"]
                    artist_name = track["artists"][0]["name"]
                else:
                    album_img = None
                    external_url = ""
                    track_name = rec["title"]
                    artist_name = rec["artist"]

                col_art, col_info = st.columns([1, 4])

                with col_art:
                    if album_img:
                        st.image(album_img, width=200)
                    else:
                        st.markdown("🎵 (no cover)")

                with col_info:
                    st.markdown(
                        """
                        <style>
                        .song-link {
                            color: white !important;
                            text-decoration: none !important;
                            transition: color 0.3s;
                        }
                        .song-link:hover {
                            color: #1db954 !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"""<h3 style='margin-bottom: 0;'>
                            <a href="{external_url}" target="_blank" class="song-link">
                            {track_name} – {artist_name}
                            </a>
                        </h3>""",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Score:** {rec['score']:.2f}")

                    with st.expander("See how your song compares"):
                        img_base64 = encode_image_to_base64(rec["radar_chart"])
                        st.markdown(
                            f"""
                            <style>
                            .radar-img {{
                                max-height: 400px;
                                width: auto;
                                display: block;
                                margin: auto;
                            }}
                            </style>
                            <img class="radar-img" src="data:image/png;base64,{img_base64}" />
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown("---")
