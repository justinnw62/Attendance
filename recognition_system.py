import cv2
import face_recognition
import numpy as np
import pickle
import os
import sqlite3
from datetime import datetime
import time

class AttendanceSystem:
    def __init__(self):
        self.known_encodings = []
        self.known_ids = []
        self.load_known_faces()
        
        self.matching_threshold = 0.4
        self.attendance_log = []
        self.last_seen = {}  # Track re-entries
        
        # Performance metrics
        self.fps = 0
        self.processing_times = []
    
    def load_known_faces(self):
        """Load all known face encodings"""
        print("📂 Loading known faces...")
        if os.path.exists('data/encodings'):
            for file in os.listdir('data/encodings'):
                if file.endswith('.pkl'):
                    student_id = file.replace('.pkl', '')
                    try:
                        with open(f'data/encodings/{file}', 'rb') as f:
                            encoding = pickle.load(f)
                            self.known_encodings.append(encoding)
                            self.known_ids.append(student_id)
                    except:
                        continue
        
        print(f"✅ Loaded {len(self.known_ids)} students")
    
    def detect_motion(self, frame, prev_frame):
        """Simple motion detection"""
        if prev_frame is None:
            return True
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # If significant motion detected
        motion_pixels = np.sum(thresh) / 255
        return motion_pixels > 500  # Adjust threshold as needed
    
    def is_human(self, frame):
        """Simple human detection using face detection"""
        # If we find faces, it's likely human
        face_locations = face_recognition.face_locations(frame)
        return len(face_locations) > 0
    
    def recognize_face(self, frame):
        """Main face recognition pipeline"""
        # 1. Face detection
        face_locations = face_recognition.face_locations(frame)
        
        if not face_locations:
            return []
        
        # 2. Face encoding
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # 3. Matching
        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            if len(self.known_encodings) == 0:
                continue
            
            # Calculate distances
            distances = face_recognition.face_distance(
                self.known_encodings, 
                face_encoding
            )
            
            # Find best match
            best_match_idx = np.argmin(distances)
            best_distance = distances[best_match_idx]
            
            if best_distance < self.matching_threshold:
                student_id = self.known_ids[best_match_idx]
                confidence = 1 - best_distance
                
                # Check if re-entry (within 5 minutes)
                is_reentry = self.check_reentry(student_id)
                
                results.append({
                    'student_id': student_id,
                    'confidence': confidence,
                    'distance': best_distance,
                    'location': face_location,
                    'is_reentry': is_reentry,
                    'timestamp': datetime.now()
                })
        
        return results
    
    def check_reentry(self, student_id):
        """Check if student re-entering within grace period"""
        current_time = time.time()
        if student_id in self.last_seen:
            time_diff = current_time - self.last_seen[student_id]
            if time_diff > 300:  # 5 minutes grace period
                return True
        
        self.last_seen[student_id] = current_time
        return False
    
    def log_attendance(self, recognition_result):
        """Log attendance to database"""
        conn = sqlite3.connect('data/attendance.db')
        cursor = conn.cursor()
        
        # Get student name
        cursor.execute('SELECT full_name FROM students WHERE student_id = ?', 
                      (recognition_result['student_id'],))
        result = cursor.fetchone()
        student_name = result[0] if result else "Unknown"
        
        # Determine status (not implementing late check for simplicity)
        status = "present" if not recognition_result['is_reentry'] else "re-entry"
        
        # Insert attendance record
        cursor.execute('''
            INSERT INTO attendance_logs (student_id, timestamp, confidence, status)
            VALUES (?, ?, ?, ?)
        ''', (
            recognition_result['student_id'],
            recognition_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            recognition_result['confidence'],
            status
        ))
        
        conn.commit()
        conn.close()
        
        return student_name, status
    
    def start_recognition(self):
        """Main recognition loop"""
        cap = cv2.VideoCapture(0)  # Mac webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        prev_frame = None
        frame_count = 0
        start_time = time.time()
        
        print("\n" + "="*50)
        print("AUTOMATED ATTENDANCE SYSTEM - RUNNING")
        print("="*50)
        print("Press 'q' to quit")
        print("Press 'm' to toggle motion detection")
        print("Press 't' to change threshold (current: 0.6)")
        print("="*50)
        
        motion_detection = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:
                self.fps = 30 / (time.time() - start_time)
                start_time = time.time()
            
            # 1. Motion Detection (optional)
            if motion_detection:
                motion_detected = self.detect_motion(frame, prev_frame)
                prev_frame = frame.copy()
                
                if not motion_detected:
                    # Display "No motion" message
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "No Motion Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Attendance System', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if self.handle_keys(key):
                        break
                    continue
            
            # 2. Human Detection
            if not self.is_human(frame):
                # Display "No human" message
                display_frame = frame.copy()
                cv2.putText(display_frame, "No Human Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imshow('Attendance System', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if self.handle_keys(key):
                    break
                continue
            
            # 3. Face Recognition
            start_process = time.time()
            recognitions = self.recognize_face(frame)
            processing_time = time.time() - start_process
            self.processing_times.append(processing_time)
            
            # Prepare display frame
            display_frame = frame.copy()
            
            # Draw results on frame
            for recognition in recognitions:
                top, right, bottom, left = recognition['location']
                
                # Draw face rectangle
                color = (0, 255, 0) if not recognition['is_reentry'] else (255, 255, 0)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                
                # Display student info
                student_name = self.get_student_name(recognition['student_id'])
                status = "RE-ENTRY" if recognition['is_reentry'] else "PRESENT"
                
                info_text = f"{recognition['student_id']}: {student_name} ({status})"
                cv2.putText(display_frame, info_text, (left, top-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Display confidence
                conf_text = f"Confidence: {recognition['confidence']:.2f}"
                cv2.putText(display_frame, conf_text, (left, bottom+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Log attendance if not re-entry
                if not recognition['is_reentry']:
                    self.log_attendance(recognition)
                    print(f"✅ {student_name} ({recognition['student_id']}) marked present")
            
            # Display system info
            self.display_system_info(display_frame)
            
            # Show the frame
            cv2.imshow('Attendance System', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if self.handle_keys(key):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        self.print_summary()
    
    def handle_keys(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            return True
        elif key == ord('m'):
            self.matching_threshold = float(input("Enter new threshold (0.4-0.7): "))
            print(f"Threshold changed to {self.matching_threshold}")
        elif key == ord('t'):
            print(f"Current threshold: {self.matching_threshold}")
        return False
    
    def get_student_name(self, student_id):
        """Get student name from database"""
        conn = sqlite3.connect('data/attendance.db')
        cursor = conn.cursor()
        cursor.execute('SELECT full_name FROM students WHERE student_id = ?', (student_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "Unknown"
    
    def display_system_info(self, frame):
        """Display system information on frame"""
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Students loaded
        cv2.putText(frame, f"Students: {len(self.known_ids)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Threshold
        cv2.putText(frame, f"Threshold: {self.matching_threshold}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (400, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    def print_summary(self):
        """Print session summary"""
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        
        # Get attendance records
        conn = sqlite3.connect('data/attendance.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.student_id, s.full_name, COUNT(*) as entries
            FROM attendance_logs a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.status != 're-entry'
            GROUP BY s.student_id
        ''')
        
        print("\n📋 Attendance Records:")
        for row in cursor.fetchall():
            print(f"   {row[0]} - {row[1]}: {row[2]} time(s)")
        
        # Performance metrics
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            print(f"\n⚡ Performance Metrics:")
            print(f"   Average processing time: {avg_time:.3f}s")
            print(f"   Total frames processed: {len(self.processing_times)}")
            print(f"   FPS: {self.fps:.1f}")
        
        conn.close()

if __name__ == "__main__":
    system = AttendanceSystem()
    system.start_recognition()