import sqlite3
import os
from datetime import datetime

DB_PATH = 'data/attendance.db'

def init_db():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/encodings'):
        os.makedirs('data/encodings')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON; ")

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
        classroom_id TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    );

    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id TEXT,
        session_date DATE,
        start_time TIME NOT NULL,
        end_time TIME NOT NULL,
        room TEXT NOT NULL,
        grace_period_minutes INTEGER DEFAULT 10,
        status TEXT DEFAULT 'scheduled',
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    );

    CREATE TABLE IF NOT EXISTS attendance_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        student_id TEXT,
        classroom_id TEXT NOT NULL,
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

    # Insert Sample Courses with HARDCODED times and classrooms
    cursor.execute("SELECT count(*) FROM courses")
    if cursor.fetchone()[0] == 0:
        # HARDCODED course schedule
        cursor.executescript('''
        INSERT INTO courses (course_id, course_name, semester) VALUES
        ('CS101', 'Introduction to Programming', 'Fall 2025'),
        ('CS102', 'Data Structures', 'Fall 2025'),
        ('BUS201', 'Business Management', 'Fall 2025');
        
        INSERT INTO sessions (course_id, session_date, start_time, end_time, room, grace_period_minutes) VALUES
        ('CS101', date('now'), '18:30', '20:00', 'Classroom 1', 10),
        ('CS102', date('now'), '18:30', '20:00', 'Classroom 2', 10),
        ('BUS201', date('now'), '18:30', '20:00', 'Classroom 3', 10);
        ''')
        
        print("✅ HARDCODED Course Schedule: ")
        print("   CS101 → Classroom 1 (18:30 - 20:00) ")
        print("   CS102 → Classroom 2 (18:30 - 20:00) ")
        print("   BUS201 → Classroom 3 (18:30 - 20:00) ")

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully. ")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_course_session_info(course_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT start_time, end_time, room, grace_period_minutes
    FROM sessions
    WHERE course_id = ?
    ORDER BY session_date DESC
    LIMIT 1
    ''', (course_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def update_sessions_to_today():
    """Update all sessions to have today's date"""
    conn = get_db_connection()
    cursor = conn.cursor()
    today = datetime.now().date()
    cursor.execute('UPDATE sessions SET session_date = ?', (today,))
    conn.commit()
    conn.close()
    print(f"✅ Sessions updated to today's date: {today}")