import os
import whisper
import streamlit as st
from pydub import AudioSegment
import pandas as pd

# Constants
audio_tags = {'comments': 'Converted using pydub!'}
upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"
fraud_keywords = [
            'Job guarantee',
            '100% placement guarantee',
            'Personal account',
            'Refund',
            'S4 Hana',
            'Server Access',
            'Free classes',
            'Lifetime Membership',
            'Providing classes in token amount',
            'Pay later',
            'Global',
            'Abusive words',
            'Sarcastic',
            'Rude',
            'Darling in ILX',
            'Freelancing support we are provided',
            'Placement support we are provided',
            'Affirm',
            'Free classes we are not provided',
            'Free Days',
            'Free trial',
            'Trial classes',
            '+ 45 Days Trial Classes'
        ]

def to_mp3(audio_file, output_audio_file):
    formats = {'wav': 'mp3', 'mp3': 'mp3', 'ogg': 'mp3', 'wma': 'mp3', 'aac': 'mp3', 'flac': 'mp3', 'flv': 'mp3', 'mp4': 'mp3'}
    extension = audio_file.name.split('.')[-1].lower()

    if extension in formats:
        audio_data = AudioSegment.from_file(os.path.join(upload_path, audio_file.name), extension)
        audio_data.export(os.path.join(download_path, output_audio_file), format=formats[extension], tags=audio_tags)
        return output_audio_file

def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file), "w") as f:
        f.write(transcript_data)

def display_audio_and_model_selection(audio_bytes):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Feel free to play your uploaded audio file 🎼")
        st.audio(audio_bytes)

    with col2:
        whisper_model_type = st.radio("Please choose your model type", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

    return whisper_model_type

def perform_fraud_detection(output_file_data):
    detected_keywords = [keyword for keyword in fraud_keywords if keyword.lower() in output_file_data.lower()]
    return len(detected_keywords) > 0, detected_keywords

def download_transcript(output_file_data, output_txt_file):
    if st.download_button(
            label="Download Transcript 📝",
            data=output_file_data,
            file_name=output_txt_file,
            mime='text/plain'
    ):
        st.balloons()
        st.success('✅ Download Successful !!')

def main():
    st.title("🗣 Automatic Speech Recognition")
    st.info('✨ Supports Audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV 😉')
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "wma", "aac", "flac", "mp4", "flv"])

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file)
        audio_file_path = os.path.join(download_path, output_audio_file)

        with st.spinner(f"Processing Audio ... 💫"):
            audio_file = open(audio_file_path, 'rb')
            audio_bytes = audio_file.read()

        st.markdown("---")
        whisper_model_type = display_audio_and_model_selection(audio_bytes)

        if st.button("Generate Transcript"):
            with st.spinner(f"Generating Transcript... 💫"):
                transcript = process_audio(str(os.path.abspath(audio_file_path)), whisper_model_type.lower())
                output_txt_file = output_audio_file.split('.')[0] + ".txt"
                save_transcript(transcript, output_txt_file)
                output_file = open(os.path.join(transcript_path, output_txt_file), "r")
                output_file_data = output_file.read()
                fraud_detected, detected_keywords = perform_fraud_detection(output_file_data)

                output_df = pd.DataFrame({
                    'Uploaded File Name': [uploaded_file.name],
                    'Output File Data': [output_file_data],
                    'Detected Keywords': [detected_keywords],
                    'Fraud Detected': [fraud_detected]
                })

                download_transcript(output_file_data, output_txt_file)

        if 'output_df' in locals():
            st.subheader("Fraud Detection Result:")
            st.write(output_df)

    else:
        st.warning('⚠ Please upload your audio file 😯')

    st.markdown(
        "<br><hr><center>Made by <a href='mailto:rigved.sarougi@henryharvin.com?subject=ASR Whisper WebApp!&body=Please specify the issue you are facing with the app.'><strong>Rigved Sarougi</strong><hr>",
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
