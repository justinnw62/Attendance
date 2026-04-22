import os
import pickle
import numpy as np
from deepface import DeepFace
import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def process_enrollment_image(img_path, student_id):
    """Extracts face embedding using DeepFace and saves as pickle."""
    try:
        # Extract embedding
        representations = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet",
            detector_backend="ssd",
            enforce_detection=True
        )
        
        if not representations:
            return False, "No face detected in image"
            
        embedding = np.array(representations[0]['embedding'])
        
        # Save to encodings folder
        os.makedirs('data/encodings', exist_ok=True)
        enc_path = f'data/encodings/{student_id}.pkl'
        with open(enc_path, 'wb') as f:
            pickle.dump(embedding, f)
            
        return True, enc_path
    except Exception as e:
        return False, str(e)

def send_attendance_email(student_id, full_name, course_code, course_name, status):
    """
    Sends attendance notification email to student.
    Email format: student_id@learner.hkuspace.hku.hk
    """
    # Construct student email
    recipient_email = f"{student_id}@learner.hkuspace.hku.hk"
    
    # Status display mapping
    status_display = {
        'present': 'Present ✅',
        'late': 'Late ⚠️',
        'absent': 'Absent ❌',
        're-entry': 'Present (Re-entry) ✅'
    }
    
    status_text = status_display.get(status, status.capitalize())
    
    # Email subject and body
    subject = f"Attendance Recorded - {course_code} ({course_name})"
    
    body = f"""Dear {full_name},

Your attendance has been successfully recorded for the following class:

📚 Course: {course_code} - {course_name}
📅 Date: {np.datetime_as_string(np.datetime64('today'), unit='D')}
✅ Status: {status_text}

If you believe this record is incorrect, please contact your instructor immediately.

Best regards,
Automated Attendance System
HKU SPACE
"""

    # Get SMTP settings from environment variables
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    
    # Check if credentials are configured
    if not sender_email or not sender_password:
        print("⚠️ Email not sent: SMTP credentials not configured in .env file")
        print("   Please add SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, and SENDER_PASSWORD to your .env file")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Attach body
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to Gmail SMTP server and send
        print(f"📧 Sending email to {recipient_email}...")
        
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()  # Upgrade connection to secure
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print(f"❌ Email authentication failed. Check your App Password in .env file")
        return False
    except smtplib.SMTPException as e:
        print(f"❌ Failed to send email to {recipient_email}: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error sending email: {str(e)}")
        return False

def test_email_connection():
    """Test if Gmail SMTP settings work correctly"""
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    
    if not sender_email or not sender_password:
        print("❌ SMTP credentials not found in .env file")
        return False
    
    try:
        print("🔧 Testing Gmail SMTP connection...")
        msg = MIMEText("This is a test email from the Attendance System.")
        msg['Subject'] = "🔧 Attendance System - Email Test"
        msg['From'] = sender_email
        msg['To'] = sender_email  # Send to yourself
        
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print("✅ Email test successful! Check your Gmail inbox.")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("❌ Authentication failed. Please check:")
        print("   1. Is 2-Factor Authentication enabled on your Gmail?")
        print("   2. Did you generate an App Password (not your regular password)?")
        print("   3. Is the App Password correctly entered in .env file?")
        return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

# Run test if this file is executed directly
if __name__ == "__main__":
    print("="*60)
    print("ATTENDANCE SYSTEM - EMAIL TEST")
    print("="*60)
    test_email_connection()