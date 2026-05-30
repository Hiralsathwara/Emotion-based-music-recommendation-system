# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import base64

# Load your dataset
df = pd.read_csv("muse_v3.csv")

# Data processing
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

# Split the data into different emotional categories
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# Function to prepare data based on emotions
def fun(emotion_list):
    data = pd.DataFrame()
    times = [30, 20, 15, 9, 2]  # Default sample sizes for different emotions

    for i, emotion in enumerate(emotion_list):
        if emotion == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=times[min(i, len(times)-1)])], ignore_index=True)
        elif emotion == 'Angry':
            data = pd.concat([data, df_angry.sample(n=times[min(i, len(times)-1)])], ignore_index=True)
        elif emotion == 'fear':
            data = pd.concat([data, df_fear.sample(n=times[min(i, len(times)-1)])], ignore_index=True)
        elif emotion == 'happy':
            data = pd.concat([data, df_happy.sample(n=times[min(i, len(times)-1)])], ignore_index=True)
        else:
            data = pd.concat([data, df_sad.sample(n=times[min(i, len(times)-1)])], ignore_index=True)

    return data

# Function to process emotions for tracking unique occurrences
def pre(emotion_list):
    emotion_counts = Counter(emotion_list)
    return list(emotion_counts.elements())

# CNN Model with transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)  # Adjust to the number of classes

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load weights if applicable
# model.load_weights('model.h5')  # Uncomment if you have pre-trained weights

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load Haarcascade Classifier
cv2.ocl.setUseOpenCL(False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit UI
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion Based Music Recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

emotion_list = []

with col2:
    if st.button('SCAN EMOTION (Click here)'):
        cap = cv2.VideoCapture(0)
        count = 0
        emotion_list.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = cv2.resize(roi_gray, (48, 48))
                cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) / 255.0  # Normalize
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))

                # Map to emotions
                emotion_list.append(max_index)  # Store index or use a dictionary for labels

                # Display the emotion on the frame
                cv2.putText(frame, str(max_index), (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
            if count >= 20:
                break
            count += 1
        
        cap.release()
        cv2.destroyAllWindows()

        # Process and recommend songs
        emotion_list = pre(emotion_list)
        st.success("Emotions successfully detected")

new_df = fun(emotion_list)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended Songs with Artist Names</b></h5>", unsafe_allow_html=True)
st.write("---------------------------------------------------------------------------------------------------------------------")

try:
    for link, artist, name in zip(new_df["link"], new_df['artist'], new_df['name']):
        st.markdown("""<h4 style='text-align: center;'><a href={}>{} - {}</a></h4>""".format(link, name, artist), unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")
except Exception as e:
    st.error("Error while displaying recommendations: {}".format(e))
