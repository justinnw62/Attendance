import sqlite3
import os

def setup_database():
    # Create database directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Connect to SQLite database
    conn = sqlite3.connect('data/attendance.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript('''
    -- Students table
    CREATE TABLE IF NOT EXISTS students (
        student_id VARCHAR(10) PRIMARY KEY,
        full_name VARCHAR(100) NOT NULL,
        programme VARCHAR(50),
        enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        encodings_path TEXT  -- Path to saved face encodings
    );
    
    -- Courses table
    CREATE TABLE IF NOT EXISTS courses (
        course_id VARCHAR(10) PRIMARY KEY,
        course_name VARCHAR(100) NOT NULL,
        semester VARCHAR(20)
    );
    
    -- Enrollment table
    CREATE TABLE IF NOT EXISTS enrollments (
        enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id VARCHAR(10),
        course_id VARCHAR(10),
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    );
    
    -- Sessions table
    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id VARCHAR(10),
        session_date DATE,
        start_time TIME,
        end_time TIME,
        room VARCHAR(20),
        status VARCHAR(20) DEFAULT 'scheduled',
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    );
    
    -- Attendance logs
    CREATE TABLE IF NOT EXISTS attendance_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        student_id VARCHAR(10),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confidence FLOAT,
        status VARCHAR(20),
        method VARCHAR(20) DEFAULT 'automatic',
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );
    
    -- Unknown faces log
    CREATE TABLE IF NOT EXISTS unknown_faces (
        face_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        encoding_path TEXT,
        matched_student_id VARCHAR(10),
        confidence FLOAT
    );
    ''')
    
    # Insert some sample data
    cursor.executescript('''
    INSERT OR IGNORE INTO students (student_id, full_name, programme) VALUES
    ('S001', 'John Doe', 'Computer Science'),
    ('S002', 'Jane Smith', 'Engineering'),
    ('S003', 'Bob Johnson', 'Business');
    
    INSERT OR IGNORE INTO courses (course_id, course_name, semester) VALUES
    ('CS101', 'Introduction to Programming', 'Fall 2025'),
    ('CS102', 'Data Structures', 'Fall 2025'),
    ('BUS201', 'Business Management', 'Fall 2025');
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database setup complete!")

if __name__ == "__main__":
    setup_database()