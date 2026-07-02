import cv2
import numpy as np
from ultralytics import YOLO

class PadelAnalysisEngine:
    def __init__(self):
        # Nano variant is highly responsive on Streamlit Cloud servers
        self.model = YOLO('yolov8n.pt') 
        self.ball_coordinates = []  
        
    def track_ball_and_players(self, frame):
        """
        Task 1: Advanced Ball and Player Filtering
        Uses tuned thresholds to catch fast-moving objects.
        """
        # Lower conf parameter slightly (0.15) to catch high-speed blurred sports balls
        results = self.model.track(frame, persist=True, classes=[0, 32], conf=0.15, verbose=False)[0]
        
        current_ball_pos = None
        boxes = results.boxes.cpu().numpy() if results.boxes is not None else []
        
        for box in boxes:
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].astype(int)
            
            # Class 32 is 'sports ball' in standard COCO schemas
            if cls == 32:
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)
                current_ball_pos = (x_center, y_center)
                self.ball_coordinates.append(current_ball_pos)
                
                # Render tracking visualization overlays
                cv2.circle(frame, (x_center, y_center), 8, (0, 255, 255), -1)
                cv2.putText(frame, "Ball", (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Class 0 is 'person'
            elif cls == 0:
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                
        return frame, current_ball_pos

    def detect_events(self, frame_idx, fps):
        """
        Task 2: Adaptive Event Handling
        Relaxed bounds to capture directional inversion even during slight frame drops.
        """
        if len(self.ball_coordinates) < 3:
            return None
            
        p1, p2, p3 = self.ball_coordinates[-3:]
        
        # Calculate vertical pixel displacements
        v1_y = p2[1] - p1[1]  
        v2_y = p3[1] - p2[1]  
        
        timestamp = f"{round(frame_idx / fps, 2)}s"
        
        # Lower thresholds (from 2 down to 0.5) to capture subtle vertical direction changes
        if v1_y > 0.5 and v2_y < -0.5:
            return {"timestamp": timestamp, "event": "Ball Bounce (Floor/Glass)"}
        elif v1_y < -0.5 and v2_y > 0.5:
            return {"timestamp": timestamp, "event": "Player Contact (Strike)"}
            
        return None

    def compile_performance_metrics(self, event_logs):
        """
        Task 3: Aggregated Report Compiler
        """
        total_events = len(event_logs)
        bounces = sum(1 for e in event_logs if "Bounce" in e['event'])
        strikes = sum(1 for e in event_logs if "Strike" in e['event'])
        
        summary_report = {
            "Match Duration Evaluated": f"{len(self.ball_coordinates)} frames monitored",
            "Total Play Events Detected": total_events,
            "Total Strikes Generated": strikes,
            "Estimated Rallies": max(1, bounces // 2),
            "Performance Status": "Analysis Complete Successfully"
        }
        return summary_report