from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import base64
import datetime
import database
import utils

app = Flask(__name__)
app.secret_key = 'super_secret_teacher_key'  # Change in production

# Ensure DB exists on start
database.init_db()

@app.route('/')
def home():
    return render_template('base.html')

# --- STUDENT PORTAL ---

@app.route('/student/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'POST':
        student_id = request.form['student_id']
        full_name = request.form['full_name']
        programme = request.form['programme']
        course_id = request.form['course_id']
        image_data = request.form['image_data']  # Base64 from JS
        
        # Save Image
        if image_data:
            header, encoded = image_data.split(',', 1)
            binary_data = base64.b64decode(encoded)
            img_path = f"data/students/{student_id}_web.jpg"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            
            with open(img_path, 'wb') as f:
                f.write(binary_data)
            
            # Generate Face Encoding
            success, result = utils.process_enrollment_image(img_path, student_id)
            
            if success:
                # Save to DB
                conn = database.get_db_connection()
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO students (student_id, full_name, programme, encodings_path)
                        VALUES (?, ?, ?, ?)
                    ''', (student_id, full_name, programme, result))
                    
                    cursor.execute('''
                        INSERT INTO enrollments (student_id, course_id)
                        VALUES (?, ?)
                    ''', (student_id, course_id))
                    
                    conn.commit()
                    return jsonify({'status': 'success', 'message': 'Enrollment successful!'})
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)})
                finally:
                    conn.close()
            else:
                return jsonify({'status': 'error', 'message': f'Face processing failed: {result}'})
        else:
            return jsonify({'status': 'error', 'message': 'No image captured'})
            
    return render_template('enroll.html')

# --- TEACHER PORTAL ---

@app.route('/teacher/login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        # Simple hardcoded password for demonstration
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
    
    # Get all students
    students = cursor.execute('SELECT * FROM students').fetchall()
    
    # Get attendance logs with student names
    logs = cursor.execute('''
        SELECT a.log_id, a.student_id, s.full_name, a.timestamp, a.confidence, a.status, a.method
        FROM attendance_logs a
        JOIN students s ON a.student_id = s.student_id
        ORDER BY a.timestamp DESC
        LIMIT 50
    ''').fetchall()
    
    conn.close()
    return render_template('dashboard.html', students=students, logs=logs)

@app.route('/teacher/manual_attendance', methods=['POST'])
def manual_attendance():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))
        
    student_id = request.form['student_id']
    status = request.form['status']  # 'present' or 'absent'
    notes = request.form.get('notes', 'Manual Entry')
    
    conn = database.get_db_connection()
    cursor = conn.cursor()
    
    # Verify student exists
    cursor.execute('SELECT student_id FROM students WHERE student_id = ?', (student_id,))
    if not cursor.fetchone():
        conn.close()
        return redirect(url_for('dashboard', error="Student not found"))
    
    # Insert Manual Log
    cursor.execute('''
        INSERT INTO attendance_logs (student_id, timestamp, confidence, status, method)
        VALUES (?, ?, ?, ?, ?)
    ''', (student_id, datetime.datetime.now(), 1.0, status, 'manual'))
    
    conn.commit()
    conn.close()
    
    return redirect(url_for('dashboard'))

@app.route('/teacher/logout')
def logout():
    session.pop('teacher_logged_in', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, port=8000)