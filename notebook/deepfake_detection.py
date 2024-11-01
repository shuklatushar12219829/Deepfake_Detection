import torch
from torch.utils.model_zoo import load_url
import matplotlib.pyplot as plt
from scipy.special import expit
import librosa
import numpy as np
import sys
from moviepy.editor import VideoFileClip
import tempfile
import streamlit as st
import os

sys.path.append('..')
from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

# Model and configuration settings
net_model = 'EfficientNetAutoAttB4'
train_db = 'DFDC'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 64

# Load model
model_url = weights.weight_url[f'{net_model}_{train_db}']
net = getattr(fornet, net_model)().eval().to(device)
net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))

transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
facedet = BlazeFace().to(device)
facedet.load_weights("../blazeface/blazeface.pth")  # Correct the path as needed
facedet.load_anchors("../blazeface/anchors.npy")  # Correct the path as needed

videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

# Function to analyze audio and determine fakeness score
def analyze_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        audio_score = np.random.rand()  # Replace with actual model inference
        return audio_score
    except Exception as e:
        st.error(f"Error in audio analysis: {e}")
        return None

# Function to process video and display results
def process_video(video_path):
    try:
        with VideoFileClip(video_path) as video_clip:
            duration = video_clip.duration
            frame_rate = video_clip.fps
            total_frames = int(duration * frame_rate)

            audio_path = extract_audio_from_video(video_clip)
            audio_score = analyze_audio(audio_path) if audio_path else None

            vid_faces = face_extractor.process_video(video_path)

            if not vid_faces:
                st.error("No faces detected in the video.")
                return

            # Ensure there are faces detected
            if len(vid_faces) == 0 or len(vid_faces[0]['faces']) == 0:
                st.error("No faces found in the video frames.")
                return

            faces_t = torch.stack([transf(image=frame['faces'][0])['image'] for frame in vid_faces if len(frame['faces']) > 0])

            with torch.no_grad():
                faces_pred = net(faces_t.to(device)).cpu().numpy().flatten()

            fake_score = expit(faces_pred.mean())

            # Display scores
            st.success(f"Video Fakeness Score: {fake_score:.4f}")
            st.text(f"Audio Fakeness Score: {audio_score:.4f}" if audio_score is not None else "Audio Fakeness Score: N/A")
            st.text(f"Duration: {duration:.2f} sec")
            st.text(f"Frame Rate: {frame_rate} FPS")
            st.text(f"Total Frames: {total_frames}")

            # Display the first detected face
            first_face_img = vid_faces[0]['faces'][0]
            st.image(first_face_img, caption="Detected Face", width=200)

    except Exception as e:
        st.error(f"Error: {e}")

def extract_audio_from_video(video_clip):
    try:
        if video_clip.audio is None:
            st.warning("The video does not contain an audio track.")
            return None
        
        audio_path = tempfile.mktemp(suffix='.wav')
        video_clip.audio.write_audiofile(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

# Streamlit application
def streamlit_app():
    st.title("Deepfake Detection System")

    uploaded_file = st.file_uploader("Upload a video (mp4/avi)", type=["mp4", "avi"])

    if uploaded_file is not None:
        st.info("Processing video and extracting audio...")

        # Write the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name
        
        audio_clip_path = None
        try:
            with VideoFileClip(temp_video_path) as video_clip:
                if video_clip.audio:
                    audio_clip_path = "temp_audio.wav"
                    video_clip.audio.write_audiofile(audio_clip_path)
                    audio_score = analyze_audio(audio_clip_path)
                    st.success("Audio extracted and processed!")
                else:
                    st.warning("The selected video has no audio.")
                    audio_score = None

            if st.button("Analyze Video"):
                st.info("Processing video frames...")
                process_video(temp_video_path)

        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if audio_clip_path and os.path.exists(audio_clip_path):
                os.remove(audio_clip_path)

if __name__ == "__main__":
    streamlit_app()
