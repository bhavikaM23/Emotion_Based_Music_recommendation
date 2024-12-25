import numpy as np
import cv2
from keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib


classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.h5")
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        output = "No Face Detected"  

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                output = emotion_labels[maxindex]

            label_position = (x, y - 10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


def detect_emotion_for_5_frames():
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise']
    emotion_mapping = {
        'Angry': 'Angry',
        'Disgust': 'Angry',   # Mapping Disgust to Angry
        'Fear': 'Calm',       # Mapping Fear to Calm
        'Happy': 'Happy',
        'Sad': 'Sad',
        'Neutral': 'Calm',    # Mapping Neutral to Calm
        'Surprise': 'Happy'   # Mapping Surprise to Happy
    }
    target_mood_mapping = {
        'Angry': 'Calm',
        'Calm': 'Happy',
        'Happy': 'Energized',
        'Sad': 'Happy'
    }

    model_path = 'FER_model.h5'  # Use the saved model path
    emotion_model = load_model(model_path)
    
    # Initialize the webcam
    camera = cv2.VideoCapture(0)  # 0 for default webcam
    img_shape = 48  # Set image dimensions for model input size
    last_detected_emotion = None  # Variable to store last detected emotion
    
    print("Recording... Press 'q' to quit after 10 frames.")

    frame_count = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image. Exiting...")
            break
        
        # Convert to grayscale for facial emotion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray_frame[y:y+h, x:x+w]
            
            # Resize to match the model input size
            resized_face = cv2.resize(face_roi, (img_shape, img_shape))
            normalized_face = resized_face / 255.0  # Rescale pixel values
            reshaped_face = np.expand_dims(normalized_face, axis=0)  # Reshape for model input
            reshaped_face = np.expand_dims(reshaped_face, axis=-1)   # Add channel dimension

            # Predict the emotion
            predictions = emotion_model.predict(reshaped_face)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_emotion = class_labels[predicted_class_index]
            
            # Update last detected emotion
            last_detected_emotion = emotion_mapping[predicted_emotion]

            # Overlay detected emotion on video frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label_position = (x, y - 10)
            cv2.putText(frame, last_detected_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the live video feed with emotions using Streamlit's st.image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB format
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Increment frame count
        frame_count += 1
        if frame_count >= 10 or cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording complete after 10 frames.")
            break

    # Release the camera
    camera.release()

    if last_detected_emotion is None:
        print("No emotion detected.")
        return None, [], []

    print(f"Last detected emotion: {last_detected_emotion}")
    target_mood = target_mood_mapping[last_detected_emotion]

    # Load the K-Means model and dataset
    kmeans_loaded = joblib.load('kmeans_model.pkl')
    loaded_df_pca = pd.read_csv('Final_Songs_dataset.csv')

    # Map last detected emotion to clusters
    filtered_by_mood = loaded_df_pca[loaded_df_pca['target_mood'] == target_mood]

    # Check if there are enough songs
    if len(filtered_by_mood) < 5:
        print("Not enough songs found for the detected emotion.")
        return last_detected_emotion, [], []

    # Select 5 random songs from the filtered DataFrame
    random_songs = filtered_by_mood.sample(5)

    # Extract the song names and URIs into separate variables
    song_names = random_songs['song_name'].tolist()
    song_uris = random_songs['uri'].tolist()

    # Print the selected songs and their URIs
    print("Selected Songs and their URIs:")
    for name, uri in zip(song_names, song_uris):
        print(f"Song: {name}, URI: {uri}")

    # Return the last detected emotion, song names, and song URIs
    return last_detected_emotion, song_names, song_uris



def main():
    # Face Analysis Application #
    st.title("Emotion Recognition and Music Recommendation")
    activities = ["Home", "Live Face Emotion Detection", "Emotion detection and song recommendation"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    # Homepage
    if choice == "Home":
        st.markdown("""
        ### Welcome to the Emotion Detection Application!
        1. Select "Live Face Emotion Detection" to analyze your emotions in real time.
        2. Use "Emotion detection and song recommendation" to capture and analyze emotions and song recommendations.
        """)
    elif choice == "Live Face Emotion Detection":
        st.header("Webcam Live Feed")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    elif choice == "Emotion detection and song recommendation":
        st.header("Emotion detection and song recommendation")
        st.write("This option will use your webcam to capture 5 frames and predict the dominant emotion.")
        if st.button("Start Detection"):
            result, song_names, song_uris = detect_emotion_for_5_frames()
            st.subheader(f"Detected Emotion: {result if result else 'No Emotion Detected'}")

            # Display songs with clickable links
            st.subheader("Recommended Songs:")
            for name, uri in zip(song_names, song_uris):
                song_link = f"""<a href="https://open.spotify.com/track/{uri.split(':')[-1]}" target="_blank" style="text-decoration: none; color: white;">{name}</a>"""
                st.markdown(song_link, unsafe_allow_html=True)

        else:
            pass


if __name__ == "__main__":
    main()
