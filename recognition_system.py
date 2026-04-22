import cv2
import numpy as np
import pickle
import os
import sqlite3
from datetime import datetime, timedelta
import time
import multiprocessing as mp
from deepface import DeepFace
from scipy.spatial.distance import cosine
import utils

class AttendanceSystem:
    def __init__(self, camera_source=0, classroom_id="Classroom 1"):
        self.camera_source = camera_source
        self.classroom_id = classroom_id
        self.known_encodings = []
        self.known_ids = []
        self.course_start_time = None
        self.grace_period_minutes = 10
        
        self.load_known_faces()
        self.load_course_schedule()
        
        # Cosine distance threshold for DeepFace/Facenet
        self.matching_threshold = 0.375
        self.attendance_log = []
        self.last_seen = {}
        self.fps = 0
        self.processing_times = []
        
        # Load Haar Cascade for fast UI bounding boxes
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _get_base_room(self):
        """Maps 'Classroom 1a' or '1b' back to 'Classroom 1' for DB queries"""
        if self.classroom_id.endswith(('a', 'b')):
            return self.classroom_id[:-1]
        return self.classroom_id

    def load_known_faces(self):
        print(f"📂 [{self.classroom_id}] Loading known faces...  ")
        self.known_encodings = []
        self.known_ids = []
        if os.path.exists('data/encodings'):
            for file in os.listdir('data/encodings'):
                if file.endswith('.pkl'):
                    student_id = file.replace('.pkl', '')
                    try:
                        with open(f'data/encodings/{file}', 'rb') as f:
                            enc = pickle.load(f)
                            self.known_encodings.append(np.array(enc))
                            self.known_ids.append(student_id)
                    except:
                         continue
        print(f"✅ [{self.classroom_id}] Loaded {len(self.known_ids)} students  ")

    def load_course_schedule(self):
        try:
            conn = sqlite3.connect('data/attendance.db')
            conn.execute("PRAGMA journal_mode=WAL;")  # Prevents locking with 3 processes
            cursor = conn.cursor()
            
            base_room = self._get_base_room()
            cursor.execute('''
                SELECT start_time, grace_period_minutes 
                FROM sessions 
                WHERE room = ? AND session_date = ?
                LIMIT 1
            ''', (base_room, datetime.now().date())) 
            result = cursor.fetchone()
            conn.close()
            
            if result:
                self.course_start_time = result[0]
                self.grace_period_minutes = result[1] or 10
                print(f"⏰ [{self.classroom_id}] Course starts at {self.course_start_time}  ")
                print(f"⏳ [{self.classroom_id}] Grace period: {self.grace_period_minutes} minutes  ")
            else:
                self.course_start_time = "09:00"
                self.grace_period_minutes = 10
                print(f"⚠️ [{self.classroom_id}] No schedule found, using default 09:00  ")
        except Exception as e:
            print(f"Error loading schedule: {e}  ")
            self.course_start_time = "09:00"
            self.grace_period_minutes = 10

    def is_student_enrolled_in_classroom(self, student_id):
        try:
            conn = sqlite3.connect('data/attendance.db')
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            
            base_room = self._get_base_room()
            cursor.execute('''
                SELECT enrollment_id FROM enrollments 
                WHERE student_id = ? AND classroom_id = ?
            ''', (student_id, base_room))
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except Exception as e:
            print(f"Error checking enrollment: {e}  ")
            return False

    def get_attendance_status(self):
        """
        Calculates status based on logic:
        - Present: From 15 mins before start to 15 mins after start.
        - Late: From 15 mins after start to 30 mins after start.
        - Absent: After 30 mins.
        - Too Early: Before 15 mins before start (Don't record).
        """
        if not self.course_start_time:
            return "error"
            
        try:
            current_time = datetime.now()
            start_hour, start_minute = map(int, self.course_start_time.split(':'))
            
            class_start_dt = current_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            
            early_threshold = class_start_dt - timedelta(minutes=15)
            present_threshold = class_start_dt + timedelta(minutes=15)
            late_threshold = class_start_dt + timedelta(minutes=30)

            if current_time < early_threshold:
                return "too_early"
            elif current_time <= present_threshold:
                return "present"
            elif current_time <= late_threshold:
                return "late"
            else:
                return "absent"
                
        except Exception as e:
            print(f"Error calculating status: {e}  ")
            return "error"

    def recognize_face(self, frame):
        if len(self.known_encodings) == 0:
            return []

        current_attendance_status = self.get_attendance_status()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
                
            try:
                reps = DeepFace.represent(img_path=face_roi, model_name="Facenet", 
                                          detector_backend="skip", enforce_detection=False)
                if not reps:
                    continue
                    
                embedding = np.array(reps[0]['embedding'])
                distances = [cosine(embedding, known_enc) for known_enc in self.known_encodings]
                best_match_idx = np.argmin(distances)
                best_distance = distances[best_match_idx]
                
                if best_distance < self.matching_threshold:
                    student_id = self.known_ids[best_match_idx]
                    confidence = 1 - best_distance
                    
                    if not self.is_student_enrolled_in_classroom(student_id):
                        print(f"⚠️ [{self.classroom_id}] {student_id} detected but NOT enrolled here - Ignored  ")
                        continue
                    
                    is_reentry = self.check_reentry(student_id)
                    status_to_log = current_attendance_status
                    final_status = "re-entry" if is_reentry else status_to_log
                    
                    location = (y, x+w, y+h, x)
                    attendance_valid = current_attendance_status not in ["too_early", "absent"]
                    
                    results.append({
                        'student_id': student_id,
                        'confidence': confidence,
                        'distance': best_distance,
                        'location': location,
                        'is_reentry': is_reentry,
                        'status': final_status,
                        'attendance_valid': attendance_valid,
                        'timestamp': datetime.now()
                    })
            except Exception as e:
                continue
                
        return results

    def get_last_attendance_time(self, student_id):
        try:
            conn = sqlite3.connect('data/attendance.db')
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            
            # Check across both doors of the same classroom to prevent duplicate logs
            base_room = self._get_base_room()
            cursor.execute('''
                SELECT timestamp FROM attendance_logs 
                WHERE student_id = ? AND classroom_id LIKE ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (student_id, f"{base_room}%"))
            
            result = cursor.fetchone()
            conn.close()
            if result:
                dt = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
                return dt.timestamp()
        except Exception as e:
            print(f"Error checking DB time: {e}  ")
        return None

    def check_reentry(self, student_id):
        current_time = time.time()
        last_db_time = self.get_last_attendance_time(student_id)
        if last_db_time and (current_time - last_db_time) < 600:  # 10 minutes
            return True
        if student_id in self.last_seen and (current_time - self.last_seen[student_id]) < 600:
            return True
        self.last_seen[student_id] = current_time
        return False

    def log_attendance(self, recognition_result):
        """Log attendance and send email notification"""
        if not recognition_result.get('attendance_valid', True):
            return None, None

        conn = sqlite3.connect('data/attendance.db')
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        
        cursor.execute('SELECT full_name FROM students WHERE student_id = ?', (recognition_result['student_id'],))
        student_row = cursor.fetchone()
        student_name = student_row[0] if student_row else "Unknown"
        
        base_room = self._get_base_room()
        cursor.execute('''
            SELECT c.course_id, c.course_name 
            FROM sessions s
            JOIN courses c ON s.course_id = c.course_id
            WHERE s.room = ? AND s.session_date = ?
            LIMIT 1
        ''', (base_room, datetime.now().date()))
        course_row = cursor.fetchone()
        course_code = course_row[0] if course_row else "N/A"
        course_name = course_row[1] if course_row else "Unknown Class"
        
        status = recognition_result['status']
        
        # Logs with specific door ID (e.g., "Classroom 1a")
        cursor.execute('''
            INSERT INTO attendance_logs (student_id, classroom_id, timestamp, confidence, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            recognition_result['student_id'],
            self.classroom_id,
            recognition_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            recognition_result['confidence'],
            status
        ))
        conn.commit()
        conn.close()
        
        utils.send_attendance_email(recognition_result['student_id'], student_name, course_code, course_name, status)
        
        return student_name, status

    def get_status_color(self, status):
        colors = { 
            'present': (0, 255, 0),
            'late': (0, 165, 255),
            're-entry': (255, 255, 0),
            'too_early': (0, 0, 255),
            'absent': (0, 0, 255),
            'unknown': (0, 0, 255)
        }
        return colors.get(status, colors['unknown'])

    def get_student_name(self, student_id):
        conn = sqlite3.connect('data/attendance.db')
        cursor = conn.cursor()
        cursor.execute('SELECT full_name FROM students WHERE student_id = ?', (student_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "Unknown"

    def start_recognition(self):
        print(f"\n📷 [{self.classroom_id}] Connecting to camera: {self.camera_source}...  ")
        cap = cv2.VideoCapture(self.camera_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"⏳ [{self.classroom_id}] Warming up camera...  ")
        time.sleep(2)
        for _ in range(10):
            cap.read()
            
        if not cap.isOpened():
            print(f"❌ [{self.classroom_id}] Failed to open camera!  ")
            return
            
        print(f"✅ [{self.classroom_id}] Camera connected successfully!  ")
        
        print(f"\n{'='*50}  ")
        print(f"AUTOMATED ATTENDANCE SYSTEM - {self.classroom_id}  ")
        print(f"{'='*50}  ")
        print(f"🕐 Course Start: {self.course_start_time}  ")
        print(f"✅ Present Window: Until {self.course_start_time} + 15 mins  ")
        print(f"️ Late Window: Until {self.course_start_time} + 30 mins  ")
        print(f"🏫 Only students enrolled in {self._get_base_room()} will be recorded  ")
        print(f"📧 Email notifications enabled  ")
        print(f"{'='*50}\n  ")
        
        frame_count = 0
        start_time = time.time()
        last_recognition_time = 0
        recognition_interval = 2.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap.open(self.camera_source)
                continue
                
            frame_count += 1
            current_time = time.time()
            
            if frame_count % 30 == 0:
                self.fps = 30 / (current_time - start_time)
                start_time = current_time
                
            if frame_count % 3 != 0:
                cv2.putText(frame, f"Room: {self.classroom_id} ", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(f'Attendance System - {self.classroom_id}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            if current_time - last_recognition_time > recognition_interval:
                recognitions = self.recognize_face(frame)
                last_recognition_time = current_time
                
                for recognition in recognitions:
                    top, right, bottom, left = recognition['location']
                    status = recognition['status']
                    color = self.get_status_color(status)
                     
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    label_bg_top = top - 30 if top > 30 else top
                    cv2.rectangle(frame, (left, label_bg_top), (right, top), color, -1)
                    
                    student_name = self.get_student_name(recognition['student_id'])
                    confidence = recognition['confidence']
                    status_display = status.upper().replace('-', ' ')
                    
                    if status == "too_early":
                        label_text = f"{student_name} | WAIT "
                    elif status == "absent":
                        label_text = f"{student_name} | ABSENT "
                    else:
                        label_text = f"{student_name} | {status_display} "
                    
                    cv2.putText(frame, label_text, (left + 5, top - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    if status != 're-entry' and recognition.get('attendance_valid', False):
                        student_name, status_str = self.log_attendance(recognition)
                        if student_name:
                            print(f"✅ [{self.classroom_id}] {student_name} ({recognition['student_id']}) - {status_str.upper()}  ")
                        
            current_status = self.get_attendance_status().upper()
            status_color = self.get_status_color(self.get_attendance_status())
            
            cv2.putText(frame, f"FPS: {self.fps:.1f} ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Students: {len(self.known_ids)} ", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {current_status} ", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Room: {self.classroom_id} ", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame,  "Press 'q' to quit ", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'Attendance System - {self.classroom_id}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n🛑 [{self.classroom_id}] System stopped.  ")

def run_classroom(camera_source, classroom_id):
    system = AttendanceSystem(camera_source=camera_source, classroom_id=classroom_id)
    system.start_recognition()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # 📷 RTSP URLs for 3 Classrooms/Doors (Update with your actual credentials)
    source1 = "rtsp://Classroom1a:12345678@192.168.0.30:554/stream1"  # 🚪 Door A
    source2 = "rtsp://Classroom1b:12345678@192.168.0.101:554/stream1"  # 🚪 Door B
    source3 = "rtsp://Classroom2:12345678@192.168.0.222:554/stream1"  # 🏫 Classroom 2

    print("="*60)
    print("STARTING MULTI-DOOR ATTENDANCE SYSTEM (DeepFace)  ")
    print("="*60)
    print("🚪 Classroom 1a (Door A) | Classroom 1b (Door B) | Classroom 2  ")
    print("⏰ Logic: Present until Start+15m -> Late until Start+30m -> Absent ")
    print("⚠️ Students enrolled in 'Classroom 1' can enter via Door A or B  ")
    print("="*60 + "\n  ")

    # 🚀 Spawn 3 independent processes
    process1 = mp.Process(target=run_classroom, args=(source1, "Classroom 1a"))
    process2 = mp.Process(target=run_classroom, args=(source2, "Classroom 1b"))
    process3 = mp.Process(target=run_classroom, args=(source3, "Classroom 2"))

    process1.start()
    process2.start()
    process3.start()

    try:
        process1.join()
        process2.join()
        process3.join()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all systems...   ")
        process1.terminate()
        process2.terminate()
        process3.terminate()
        process1.join()
        process2.join()
        process3.join()
        print("✅ All systems stopped.   ")