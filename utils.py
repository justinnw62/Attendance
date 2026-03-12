import face_recognition
import pickle
import numpy as np
import os

def process_enrollment_image(image_path, student_id):
    """
    Loads an image, generates face encoding, and saves it as .pkl
    to remain compatible with recognition_system.py
    """
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            # Save the first face encoding found
            encoding_path = f"data/encodings/{student_id}.pkl"
            with open(encoding_path, 'wb') as f:
                pickle.dump(encodings[0], f)
            return True, encoding_path
        else:
            return False, "No face detected in image"
    except Exception as e:
        return False, str(e)