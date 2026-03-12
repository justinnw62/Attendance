import sqlite3
import os

DB_PATH = 'data/attendance.db'

def init_db():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/encodings'):
        os.makedirs('data/encodings')
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable Foreign Keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Create Tables (Corrected Syntax)
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        programme TEXT,
        enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        encodings_path TEXT
    );

    CREATE TABLE IF NOT EXISTS courses (
        course_id TEXT PRIMARY KEY,
        course_name TEXT NOT NULL,
        semester TEXT
    );

    CREATE TABLE IF NOT EXISTS enrollments (
        enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        course_id TEXT,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    );

    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id TEXT,
        session_date DATE,
        start_time TIME,
        end_time TIME,
        room TEXT,
        status TEXT DEFAULT 'scheduled',
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    );

    CREATE TABLE IF NOT EXISTS attendance_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        student_id TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confidence REAL,
        status TEXT,
        method TEXT DEFAULT 'automatic',
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (student_id) REFERENCES students(student_id) 
    );

    CREATE TABLE IF NOT EXISTS unknown_faces (
        face_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        encoding_path TEXT,
        matched_student_id TEXT,
        confidence REAL
    );
    ''')
    
    # Insert Sample Courses if empty
    cursor.execute("SELECT count(*) FROM courses")
    if cursor.fetchone()[0] == 0:
        cursor.executescript('''
        INSERT INTO courses (course_id, course_name, semester) VALUES
        ('CS101', 'Introduction to Programming', 'Fall 2025'),
        ('CS102', 'Data Structures', 'Fall 2025'),
        ('BUS201', 'Business Management', 'Fall 2025');
        ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn