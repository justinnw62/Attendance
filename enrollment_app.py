import cv2
import face_recognition
import numpy as np
import os
import pickle
import sqlite3
from datetime import datetime

#For recording student data into the database

class EnrollmentSystem:
    def __init__(self):
        self.known_encodings = []
        self.known_ids = []
        self.angle_names = [
            "Front View",
            "15° Left Turn",
            "15° Right Turn",
            "30° Left Turn",
            "30° Right Turn",
            "Upward Tilt",
            "Downward Tilt",
            "Slight Left Profile",
            "Slight Right Profile",
            "Neutral Expression"
        ]
        
        # Create directories
        if not os.path.exists('data/students'):
            os.makedirs('data/students')
        if not os.path.exists('data/encodings'):
            os.makedirs('data/encodings')
    
    def capture_images(self, student_id, student_name):
        """Capture 10 images from different angles"""
        cap = cv2.VideoCapture(0)  # Use Mac webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        images = []
        current_angle = 0
        
        print(f"\n📸 Starting enrollment for {student_name} ({student_id})")
        print("=" * 50)
        
        while len(images) < 10:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture image")
                break
            
            # Display instructions
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Angle {len(images)+1}/10: {self.angle_names[len(images)]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Show face detection box
            face_locations = face_recognition.face_locations(frame)
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            cv2.imshow('Enrollment - Follow Instructions', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space bar to capture
                if face_locations:
                    # Save the image
                    save_path = f"data/students/{student_id}_angle_{len(images)+1}.jpg"
                    cv2.imwrite(save_path, frame)
                    images.append(save_path)
                    print(f"✅ Captured: {self.angle_names[len(images)-1]}")
                    
                    # Show confirmation
                    confirmation_frame = frame.copy()
                    cv2.putText(confirmation_frame, "CAPTURED!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Enrollment - Follow Instructions', confirmation_frame)
                    cv2.waitKey(500)  # Show confirmation for 500ms
                else:
                    print("❌ No face detected. Please position face in frame.")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return images
    
    def process_images(self, student_id, image_paths):
        """Generate face encodings from captured images"""
        encodings = []
        
        print("\n🔄 Processing images...")
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = face_recognition.load_image_file(image_path)
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    encodings.append(face_encodings[0])
                    print(f"✅ Angle {i+1}: Encoding generated")
                else:
                    print(f"❌ Angle {i+1}: No face found")
                    
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
        
        if encodings:
            # Save encodings
            avg_encoding = np.mean(encodings, axis=0)
            encoding_path = f"data/encodings/{student_id}.pkl"
            
            with open(encoding_path, 'wb') as f:
                pickle.dump(avg_encoding, f)
            
            # Save to database
            self.save_to_database(student_id, encoding_path)
            
            print(f"\n✅ Enrollment complete for student {student_id}")
            print(f"   Images: {len(image_paths)}")
            print(f"   Encodings saved to: {encoding_path}")
            
            return True
        
        return False
    
    def save_to_database(self, student_id, encoding_path):
        """Save student info to database"""
        conn = sqlite3.connect('data/attendance.db')
        cursor = conn.cursor()
        
        # Update student record with encoding path
        cursor.execute('''
            UPDATE students 
            SET encodings_path = ?
            WHERE student_id = ?
        ''', (encoding_path, student_id))
        
        conn.commit()
        conn.close()
    
    def enroll_student(self):
        """Main enrollment function"""
        print("\n" + "="*50)
        print("STUDENT ENROLLMENT SYSTEM")
        print("="*50)
        
        # Get student info
        student_id = input("Enter Student ID (e.g., S004): ").strip()
        student_name = input("Enter Full Name: ").strip()
        programme = input("Enter Programme: ").strip()
        
        # Save to database
        conn = sqlite3.connect('data/attendance.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO students (student_id, full_name, programme)
            VALUES (?, ?, ?)
        ''', (student_id, student_name, programme))
        conn.commit()
        conn.close()
        
        # Capture images
        print("\n" + "-"*50)
        print("IMAGE CAPTURE INSTRUCTIONS:")
        print("1. Sit 1 meter from camera")
        print("2. Ensure good lighting")
        print("3. Face should be clearly visible")
        print("4. Follow on-screen prompts for angles")
        print("-"*50)
        input("Press Enter when ready to start camera...")
        
        images = self.capture_images(student_id, student_name)
        
        if len(images) >= 8:  # Minimum 8 images
            success = self.process_images(student_id, images)
            if success:
                print(f"\n🎉 Enrollment successful! Student {student_id} is now in the system.")
        else:
            print(f"\n❌ Enrollment failed. Only captured {len(images)} images. Need at least 8.")

if __name__ == "__main__":
    enrollment = EnrollmentSystem()
    enrollment.enroll_student()