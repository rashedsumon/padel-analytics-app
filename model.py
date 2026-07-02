import cv2
import numpy as np
from ultralytics import YOLO

class PadelAnalysisEngine:
    def __init__(self):
        # Using an optimized pre-trained YOLO Nano variant for fast execution in Streamlit Cloud
        self.model = YOLO('yolov8n.pt') 
        self.ball_coordinates = []  # Stores stream of 2D [x, y] coordinates
        
    def track_ball_and_players(self, frame):
        """
        Task 1: Ball and Player Detection
        Accepts a video frame and detects bounding boxes/coordinates.
        """
        # Inference using YOLO (tracking class 0 for person, class 32 for sports ball)
        results = self.model.track(frame, persist=True, classes=[0, 32], verbose=False)[0]
        
        current_ball_pos = None
        boxes = results.boxes.cpu().numpy() if results.boxes is not None else []
        
        for box in boxes:
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].astype(int)
            
            # If a ball is detected, extract its center center [x, y]
            if cls == 32:
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)
                current_ball_pos = (x_center, y_center)
                self.ball_coordinates.append(current_ball_pos)
                
                # Render visual trail on the frame
                cv2.circle(frame, (x_center, y_center), 7, (0, 255, 255), -1)
            
            # Render Player bounding box
            elif cls == 0:
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                
        return frame, current_ball_pos

    def detect_events(self, frame_idx, fps):
        """
        Task 2: Event Detection (Bounces & Hits)
        Analyzes directional shifts in historical trajectories to label events.
        """
        if len(self.ball_coordinates) < 3:
            return None
            
        # Extract the last 3 positions to evaluate vector changes
        p1, p2, p3 = self.ball_coordinates[-3:]
        
        # Calculate directional vectors
        v1_y = p2[1] - p1[1]  # Vector 1 Y-axis change
        v2_y = p3[1] - p2[1]  # Vector 2 Y-axis change
        
        timestamp = str(np.round(frame_idx / fps, 2)) + "s"
        
        # Physics Rule Engine: Sharp Y-axis inversion implies a bounce or racket contact
        if v1_y > 2 and v2_y < -2:
            return {"timestamp": timestamp, "event": "Ball Bounce / Floor-Glass Interaction"}
        elif v1_y < -2 and v2_y > 2:
            return {"timestamp": timestamp, "event": "Player Strike / Smash / Volley"}
            
        return None

    def compile_performance_metrics(self, event_logs):
        """
        Task 3: High-Level Performance Metrics Dashboard Summary
        """
        total_events = len(event_logs)
        bounces = sum(1 for e in event_logs if "Bounce" in e['event'])
        strikes = sum(1 for e in event_logs if "Strike" in e['event'])
        
        # Generate a structured data report summary
        summary_report = {
            "Match Duration Evaluated": f"{len(self.ball_coordinates)} processed steps",
            "Total Play Events Detected": total_events,
            "Total Strikes Generated": strikes,
            "Estimated Rallies": bounces // 2 if bounces > 0 else 1,
            "Performance Status": "Analysis Complete"
        }
        return summary_report