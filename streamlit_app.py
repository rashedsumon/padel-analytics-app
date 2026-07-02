import streamlit as tf
import cv2
import numpy as np
import tempfile
import pandas as pd
from model import PadelAnalysisEngine
from data_loader import load_datasets

# Initialize UI Streamlit Configuration
st.set_page_config(page_title="AI Padel Performance Analyzer", layout="wide")
st.title("🎾 AI Padel Performance Tracking & Analytics Dashboard")

# Background pipeline configuration to keep Streamlit cloud environments fast
@st.cache_resource
def trigger_data_sync():
    return load_datasets()

# Run background dataset loader
trigger_data_sync()

engine = PadelAnalysisEngine()

uploaded_file = st.sidebar.file_uploader("Upload a Padel Match Video File (.mp4)", type=["mp4"])

if uploaded_file is not None:
    # Save file temporarily to disk to allow OpenCV streaming
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    video_capture = cv2.VideoCapture(tfile.name)
    fps = video_capture.get(cv2.CAP_PROP_FPS) or 30.0
    
    st.sidebar.success("Video Loaded Successfully!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1 & 2. Ball Tracking & Event Log Stream")
        video_placeholder = st.empty()
        
    with col2:
        st.subheader("Live Event Tracker Log")
        event_placeholder = st.empty()
        
    event_logs = []
    frame_idx = 0
    
    # Process Video Frames
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 360)) # Optimize output for screen canvas resizing
        
        # 1. Run tracking models
        processed_frame, ball_coords = engine.track_ball_and_players(frame)
        
        # 2. Extract Event Detections
        detected_event = engine.detect_events(frame_idx, fps)
        if detected_event:
            event_logs.append(detected_event)
            df_logs = pd.DataFrame(event_logs).tail(10)
            event_placeholder.dataframe(df_logs, use_container_width=True)
            
        # Update Web view component frame stream
        video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
        frame_idx += 1
        
    video_capture.release()
    
    # 3. Print out final Performance Summary Report
    st.markdown("---")
    st.subheader("3. Final Compiled High-Level Performance Metrics Summary")
    metrics_report = engine.compile_performance_metrics(event_logs)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Events Logged", metrics_report["Total Play Events Detected"])
    m_col2.metric("Total Strikes Hit", metrics_report["Total Strikes Generated"])
    m_col3.metric("Estimated Rallies Played", metrics_report["Estimated Rallies"])
    m_col4.metric("Engine Execution State", metrics_report["Performance Status"])
    
    st.json(metrics_report)
else:
    st.info("Please upload an MP4 match recording from the left sidebar panel to initialize the tracking framework.")