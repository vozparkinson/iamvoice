import os
import tempfile
import numpy as np
import librosa
import scipy.spatial.distance
from fastdtw import fastdtw
import streamlit as st


def load_and_preprocess(file_input, sr=22050):
    """
    Loads audio from a file path or file-like object, converts to mono,
    trims silence, and normalizes audio.
    """
    # Load audio (Librosa/Soundfile automatically handles wav, mp3, ogg, flac, etc.)
    y, sr = librosa.load(file_input, sr=sr, mono=True)

    # Trim leading and trailing silence (therapy recordings often have silence gaps)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    # Audio normalization to equalize volume variations
    if np.max(np.abs(y_trimmed)) > 0:
        y_trimmed = y_trimmed / np.max(np.abs(y_trimmed))

    return y_trimmed, sr


def extract_mfcc(y, sr, n_mfcc=13):
    """Extracts MFCC features along with delta features for temporal dynamics."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Delta features capture changes in speech over time
    delta_mfcc = librosa.feature.delta(mfcc)

    # Stack static and dynamic features
    features = np.vstack([mfcc, delta_mfcc])

    # Z-score normalization per feature band
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True) + 1e-8
    normalized_features = (features - mean) / std

    return normalized_features.T


def compute_similarity(file1, file2, scale_factor=35.0):
    """
    Computes DTW distance between two audio inputs and maps it to a 0-100% similarity score.
    scale_factor: Adjusts the decay rate (30-40 works well for MFCC+Delta voice comparisons).
    """
    y1, sr1 = load_and_preprocess(file1)
    y2, sr2 = load_and_preprocess(file2)

    feat1 = extract_mfcc(y1, sr1)
    feat2 = extract_mfcc(y2, sr2)

    # Compute Dynamic Time Warping distance between frame feature vectors
    distance, _ = fastdtw(feat1, feat2, dist=scipy.spatial.distance.cosine)

    # Normalize DTW distance by average length to prevent longer recordings from getting unfair high distances
    avg_len = (len(feat1) + len(feat2)) / 2.0
    norm_distance = distance / avg_len

    # Exponential mapping to 0-100% scale
    similarity_pct = 100.0 * np.exp(-norm_distance * (100.0 / scale_factor))
    return np.clip(similarity_pct, 0.0, 100.0), norm_distance


# --- STREAMLIT UI ---
def run_app():
    st.set_page_config(page_title="Voice Matcher", layout="centered")

    st.title("🗣️ Voice Similarity Analyzer")
    st.write(
        "Upload a target voice sample (e.g., therapist) and a patient sample to compare timbre and articulation profile."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Artist Recording")
        audio_file1 = st.file_uploader(
            "Upload Reference Audio",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            key="ref",
        )
        if audio_file1:
            st.audio(audio_file1)

    with col2:
        st.subheader("Your Recording")
        audio_file2 = st.file_uploader(
            "Upload Patient Audio",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            key="pat",
        )
        if audio_file2:
            st.audio(audio_file2)

    if audio_file1 and audio_file2:
        if st.button("Analyze & Compare", type="primary", use_container_width=True):
            with st.spinner("Processing audio features and running DTW alignment..."):
                # Save bytes to temporary files for librosa processing
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(audio_file1.name)[1]
                ) as tmp1, tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(audio_file2.name)[1]
                ) as tmp2:

                    tmp1.write(audio_file1.read())
                    tmp2.write(audio_file2.read())
                    tmp1_path = tmp1.name
                    tmp2_path = tmp2.name

                try:
                    score, raw_dist = compute_similarity(tmp1_path, tmp2_path)

                    st.markdown("---")
                    st.metric(
                        label="Voice Match Score", value=f"{score:.1f}%"
                    )

                    st.progress(score / 100.0)

                    if score >= 80:
                        st.success("High match: Strong similarity in speech spectral profile.")
                    elif score >= 50:
                        st.info("Moderate match: Similar characteristics, with notable variations.")
                    else:
                        st.warning("Low match: Significant variation in articulation or pitch profile.")

                    with st.expander("Technical Metrics"):
                        st.write(f"Normalized DTW Distance: `{raw_dist:.4f}`")

                finally:
                    # Clean up temp files
                    if os.path.exists(tmp1_path):
                        os.remove(tmp1_path)
                    if os.path.exists(tmp2_path):
                        os.remove(tmp2_path)


if __name__ == "__main__":
    run_app()
