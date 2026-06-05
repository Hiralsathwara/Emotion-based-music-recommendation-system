import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, filename="error_log.log", filemode="a")

# Load dataset and preprocess
df = pd.read_csv("muse_v3.csv")
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]

# Remove rows with Spotify links
df = df[~df['link'].str.contains('spotify', na=False)]

# Sort the dataset by emotional and pleasant tags and reset index for proper handling
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index(drop=True, inplace=True)

# Split data for emotion-based categories
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# Enhanced Emotion-to-Song Mapping and Dynamic Song Recommendation Based on Emotion Intensity
def fun(list_emotions, emotion_intensity):
    data = pd.DataFrame()
    sample_size = [50, 40, 30, 20] if emotion_intensity == 'high' else [30, 20, 15, 10]

    for idx, emotion in enumerate(list_emotions):
        if emotion == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=min(sample_size[idx], len(df_neutral)))], ignore_index=True)
        elif emotion == 'Angry':
            data = pd.concat([data, df_angry.sample(n=min(sample_size[idx], len(df_angry)))], ignore_index=True)
        elif emotion == 'Fear':
            data = pd.concat([data, df_fear.sample(n=min(sample_size[idx], len(df_fear)))], ignore_index=True)
        elif emotion == 'Happy':
            data = pd.concat([data, df_happy.sample(n=min(sample_size[idx], len(df_happy)))], ignore_index=True)
        else:
            data = pd.concat([data, df_sad.sample(n=min(sample_size[idx], len(df_sad)))], ignore_index=True)
    return data

def pre(list_emotions):
    return list(Counter(list_emotions).keys())

def optimize_pipeline(list_emotions, emotion_intensity):
    results = []
    if list_emotions:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fun, [emotion], emotion_intensity) for emotion in list_emotions]
            for future in futures:
                result = future.result()
                if not result.empty:
                    results.append(result)
    else:
        logging.error("No emotions detected to generate recommendations.")
    
    return pd.concat(results) if results else pd.DataFrame()

def real_time_playlist_update(list_emotions, emotion_intensity):
    updated_playlist = optimize_pipeline(list_emotions, emotion_intensity)
    return updated_playlist

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

try:
    model.load_weights('model.h5')
except ValueError as e:
    logging.error(f"Error loading weights: {str(e)}")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Custom CSS for colors and styling
st.markdown(
    """
    <style>
    body { background-color: #eef2f3; font-family: 'Arial', sans-serif; }
    .stButton>button {
        color: white;
        background-color: #6a1b9a;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #4a148c; }
    .emotion-title {
        color: #283593;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .song-recommendation {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    .song-recommendation h4 { color: #1a237e; margin: 5px; }
    .song-recommendation h4 a { text-decoration: none; }
    .song-recommendation p { color: #3949ab; margin: 5px; font-style: italic; }
    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h2 class='emotion-title'>🎵 Emotion Based Music Recommendation 🎵</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
list_emotions = []
recommended_songs = pd.DataFrame()

# Real-time emotion detection button
with col2:
    if st.button('🎥 Scan Emotion'):
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])
        while len(list_emotions) < 20:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[max_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                list_emotions.append(emotion_dict[max_index])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
        list_emotions = pre(list_emotions)
        cap.release()
        cv2.destroyAllWindows()

# Emotion intensity selection
emotion_intensity = st.selectbox('💡 Select Emotion Intensity:', ['normal', 'high'])

# Generate recommendations
if list_emotions:
    recommended_songs = real_time_playlist_update(list_emotions, emotion_intensity)
    if recommended_songs.empty:
        st.write("No songs available for the detected emotions.")
else:
    st.write("No emotions detected. Please try scanning again.")

# Display recommended songs
if not recommended_songs.empty:
    st.markdown("<h3 class='emotion-title'>🎧 Recommended Songs:</h3>", unsafe_allow_html=True)
    for _, row in recommended_songs.iterrows():
        st.markdown(
            f"""
            <div class='song-recommendation'>
                <h4><a href='{row['link']}' target='_blank'>{row['name']}</a></h4>
                <p>{row['artist']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown("<div class='footer'>Powered by 🎵 Group 18</div>", unsafe_allow_html=True)
