import streamlit as st
# import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder, speech_to_text
import numpy as np
import pickle  # Assuming your model is saved as a pickle file
import time

# Load your pre-trained fraud detection model
# Replace 'model.pkl' with your actual model file
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Function to record audio and convert to text
def record_and_transcribe():
    # recognizer = sr.Recognizer()
    # for index, name in enumerate(sr.Microphone.list_microphone_names()):
    #     print(f"Microphone with name {name} found for Microphone(device_index={index})")
    #     st.write(f"Microphone with name {name} found for Microphone(device_index={index})")

    # with sr.Microphone(device_index=2) as source:
    #     st.info("Listening... Speak into the microphone.")
    #     try:
    #         audio = recognizer.listen(source, timeout=5)
    #         text = recognizer.recognize_google(audio)
    #         return text
    #     except sr.UnknownValueError:
    #         return "Unable to understand the speech."
    #     except sr.RequestError:
    #         return "Speech recognition service is unavailable."

    def callback():
        if st.session_state.my_stt_output:
            st.write(st.session_state.my_stt_output)

    speech_to_text(key='my_stt', callback=callback)


# Fraud detection function
def detect_fraud(text):
    # Preprocess the text to match your model's input requirements
    # Replace this with actual preprocessing
    processed_text = np.array([len(text)])  # Example: using text length as a feature
    prediction = model.predict([processed_text])
    return "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"

# Streamlit app layout
st.title("Real-Time Fraud Detection via Speech")
st.write("Speak into your microphone, and we'll determine if the content is fraudulent.")

if st.button("Start"):
    st.write("Listening...")
    speech_text = record_and_transcribe()
    
    if speech_text:
        st.write(f"Transcribed Text: {speech_text}")
        with st.spinner("Analyzing for fraud..."):
            result = detect_fraud(speech_text)
            time.sleep(2)  # Simulate processing time
        st.success(f"Fraud Detection Result: {result}")

st.write("This app uses speech recognition and machine learning to detect fraud in real time.")

