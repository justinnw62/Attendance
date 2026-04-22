from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import os
import base64
import datetime
import database
import utils
import sqlite3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'super_secret_teacher_key'

# Initialize database
database.init_db()

def update_sessions_to_today():
    """Update all sessions to today's date"""
    conn = sqlite3.connect('data/attendance.db')
    cursor = conn.cursor()
    today = datetime.datetime.now().date()
    cursor.execute('UPDATE sessions SET session_date = ?', (today,))
    conn.commit()
    conn.close()

# Run date update
update_sessions_to_today()

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/student/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'POST':
        student_id = request.form['student_id']
        full_name = request.form['full_name']
        programme = request.form['programme']
        course_id = request.form['course_id']
        image_data = request.form['image_data']
        
        classroom_mapping = {
            'CS101': 'Classroom 1',
            'CS102': 'Classroom 2',
            'BUS201': 'Classroom 3'
        }
        
        assigned_classroom = classroom_mapping.get(course_id, 'Classroom 1')
        
        session_info = database.get_course_session_info(course_id)
        start_time = session_info['start_time'] if session_info else '09:00'
        end_time = session_info['end_time'] if session_info else '11:00'
        
        if image_data:
            header, encoded = image_data.split(',', 1)
            binary_data = base64.b64decode(encoded)
            img_path = f"data/students/{student_id}_web.jpg"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            
            with open(img_path, 'wb') as f:
                f.write(binary_data)
            
            success, result = utils.process_enrollment_image(img_path, student_id)
            
            if success:
                conn = database.get_db_connection()
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO students (student_id, full_name, programme, encodings_path)
                        VALUES (?, ?, ?, ?)
                    ''', (student_id, full_name, programme, result))
                    
                    cursor.execute('''
                        INSERT INTO enrollments (student_id, course_id, classroom_id)
                        VALUES (?, ?, ?)
                    ''', (student_id, course_id, assigned_classroom))
                    
                    conn.commit()
                    return jsonify({
                        'status': 'success', 
                        'message': 'Enrollment successful!',
                        'classroom': assigned_classroom,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)})
                finally:
                    conn.close()
            else:
                return jsonify({'status': 'error', 'message': f'Face processing failed: {result}'})
        else:
            return jsonify({'status': 'error', 'message': 'No image captured'})
            
    return render_template('enroll.html')

@app.route('/teacher/login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        if request.form['password'] == 'teacher123':
            session['teacher_logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('teacher_login.html', error="Invalid Password")
    return render_template('teacher_login.html')

@app.route('/teacher/dashboard')
def dashboard():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))
    
    conn = database.get_db_connection()
    cursor = conn.cursor()

    students = cursor.execute('SELECT * FROM students').fetchall()

    courses = cursor.execute('''
        SELECT c.course_id, c.course_name, s.start_time, s.end_time, s.room, s.grace_period_minutes
        FROM courses c
        LEFT JOIN sessions s ON c.course_id = s.course_id
        WHERE s.session_date = ? OR s.session_date IS NULL
        ORDER BY s.start_time
    ''', (datetime.datetime.now().date(),)).fetchall()

    logs = cursor.execute('''
        SELECT a.log_id, a.student_id, s.full_name, a.timestamp, a.confidence, a.status, a.method, a.classroom_id
        FROM attendance_logs a
        JOIN students s ON a.student_id = s.student_id
        ORDER BY a.timestamp DESC
        LIMIT 50
    ''').fetchall()

    conn.close()
    return render_template('dashboard.html', students=students, logs=logs, courses=courses)

@app.route('/teacher/manual_attendance', methods=['POST'])
def manual_attendance():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))
    
    student_id = request.form['student_id']
    status = request.form['status']
    classroom = request.form.get('classroom', 'Classroom 1')

    if not student_id or not status:
        flash('Please select a student and status', 'error')
        return redirect(url_for('dashboard'))

    conn = database.get_db_connection()
    cursor = conn.cursor()

    # Get student info
    cursor.execute('SELECT student_id, full_name FROM students WHERE student_id = ?', (student_id,))
    student_row = cursor.fetchone()
    if not student_row:
        conn.close()
        flash('Student not found', 'error')
        return redirect(url_for('dashboard'))
    student_name = student_row['full_name']

    # Get course info for this classroom today
    cursor.execute('''
        SELECT c.course_id, c.course_name 
        FROM sessions s
        JOIN courses c ON s.course_id = c.course_id
        WHERE s.room = ? AND s.session_date = ?
        LIMIT 1
    ''', (classroom, datetime.datetime.now().date()))
    course_row = cursor.fetchone()
    course_code = course_row['course_id'] if course_row else "N/A"
    course_name = course_row['course_name'] if course_row else "Unknown Class"

    # Insert attendance record
    cursor.execute('''
        INSERT INTO attendance_logs (student_id, classroom_id, timestamp, confidence, status, method)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (student_id, classroom, datetime.datetime.now(), 1.0, status, 'manual'))

    conn.commit()
    conn.close()

    # Send email notification
    utils.send_attendance_email(student_id, student_name, course_code, course_name, status)

    flash(f'Attendance recorded for student {student_id} as {status}. Email sent!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/teacher/logout')
def logout():
    session.pop('teacher_logged_in', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, port=1030)