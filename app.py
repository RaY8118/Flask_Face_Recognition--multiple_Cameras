from flask import Flask, render_template, Response, flash, request, redirect, url_for, session, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import asc
from werkzeug.utils import secure_filename
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.exceptions import NotFound
from sqlalchemy.exc import SQLAlchemyError
import cv2
import pickle
import numpy as np
import face_recognition
import cvzone
import datetime
from datetime import time as datetime_time
import time
import threading
import os
import csv
import io
import logging
import json
import re


# Opening all the necessary files needed
with open('config.json') as p:
    params = json.load(p)['params']
encoding_file_path = params['encoding_file_path']
file = open(encoding_file_path, 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds


# App configs
app = Flask(__name__)
app.config['SECRET_KEY'] = params['secret_key']
app.config['SQLALCHEMY_DATABASE_URI'] = params['sql_url']
db = SQLAlchemy(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = params['upload_folder']
hostedapp = Flask(__name__)
hostedapp.wsgi_app = DispatcherMiddleware(
    NotFound(), {"/Attendance_system": app})
cert_path = params['cert_path']
key_path = params['key_path']
bcrypt = Bcrypt()
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)


# Variables defined
camera = None  # Global variable to store camera object
morn_time = datetime_time(int(params['morning_time']))
even_time = datetime_time(int(params['evening_time']))
curr_time = datetime.datetime.now().time()


# Logic to find what function to call based on the time of day for marking the attendance
if morn_time <= curr_time < even_time:
    morn_attendance = True
    even_attendance = False
else:
    even_attendance = True
    morn_attendance = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Models used to connect in SQL Alchemy
# Model of students data table
class Student_data(db.Model):
    __tablename__ = 'student_data'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    rollno = db.Column(db.String(120), unique=True, nullable=False)
    division = db.Column(db.String(80), nullable=False)
    branch = db.Column(db.String(80), nullable=False)
    regid = db.Column(db.String(80), unique=True, nullable=False)


# Model of Attendance table
class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.String(20))
    end_time = db.Column(db.String(20))
    date = db.Column(db.Date, default=datetime.date.today)
    roll_no = db.Column(db.String(20), nullable=False, unique=False)
    division = db.Column(db.String(10))
    branch = db.Column(db.String(100))
    reg_id = db.Column(db.String(100))


# Model of users table
class Users(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    reg_id = db.Column(db.String(20), nullable=False)
    role = db.Column(db.String(20))

    def __repr__(self):
        return f'<User: {self.username}, Role: {self.role}>'

    def get_id(self):
        return str(self.id)


@login_manager.user_loader
def load_user(user_id):
    return db.session.query(Users).get(int(user_id))


# Function to start the camera
def start_camera():
    global camera


# Function to stop the camera
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None


# Function for comparing incoming face with encoded file
def compare(encodeListKnown, encodeFace):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    # print("matches", matches)
    # print("faceDis", faceDis)
    matchIndex = np.argmin(faceDis)
    return matches, faceDis, matchIndex


# Function to get name of student from the index given by comparing function
def get_data(matches, matchIndex, studentIds):
    if matches[matchIndex]:
        student_id = studentIds[matchIndex]  # ID from face recognition
        return student_id
    return None  # Return None if no match found


# Function which writes the morning attendance in the database
def morningattendance(name, current_date, roll_no, div, branch, reg_id):
    time.sleep(3)
    try:
        with app.app_context():
            existing_entry = Attendance.query.filter(
                Attendance.name == name,
                Attendance.date == current_date,
                Attendance.start_time != None
            ).first()

            if existing_entry:
                print("Your Attendance is already recorded before")
            else:
                new_attendance = Attendance(
                    name=name,
                    start_time=datetime.datetime.now().strftime("%H:%M:%S"),
                    date=current_date,
                    roll_no=roll_no,
                    division=div,
                    branch=branch,
                    reg_id=reg_id
                )
                db.session.add(new_attendance)
                db.session.commit()
                print("Start time and student data recorded in the database")
    except Exception as e:
        print("Error:", e)


# Function which writes the evening attendance in the database
def eveningattendance(name, current_date):
    time.sleep(3)
    try:
        with app.app_context():
            existing_entry = Attendance.query.filter(
                Attendance.name == name,
                Attendance.date == current_date,
                Attendance.start_time != None
            ).first()
            recorded_entry = Attendance.query.filter(
                Attendance.name == name,
                Attendance.end_time != None
            ).first()

            if existing_entry and not recorded_entry:
                existing_entry.end_time = datetime.datetime.now().strftime("%H:%M:%S")
                db.session.commit()
                print("End time recorded in the database")
            elif recorded_entry:
                print("End time already recorded!")
            else:
                print("No existing entry found for evening attendance")
    except Exception as e:
        print("Error:", e)


# Function which gets data of identified student from the database
def mysqlconnect(student_id):
    # If student_id is None, return None for all values
    if student_id is None:
        return None, None, None, None, None

    try:
        with app.app_context():
            # Query student data using SQLAlchemy
            student_data = Student_data.query.filter_by(
                regid=student_id).first()

            if student_data:
                # If student data is found, extract values
                id = student_data.id
                name = student_data.name
                roll_no = student_data.rollno
                division = student_data.division
                branch = student_data.branch

                return id, name, roll_no, division, branch
            else:
                # If no student is found, return None for all values
                return None, None, None, None, None

    except Exception as e:
        print("Error:", e)
        return None, None, None, None, None


# Function which does the face recognition and displaying the video feed
def gen_frames(camera):
    while camera is not None:
        success, frame = camera.read()
        if not success:
            break
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches, facedis, matchIndex = compare(encodeListKnown, encodeFace)
            student_id = get_data(matches, matchIndex, studentIds)
            data = mysqlconnect(student_id)
            name = data[1]
            roll_no = data[2]
            div = data[3]
            branch = data[4]
            reg_id = student_id
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = x1, y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(frame, bbox, rt=0)
            cv2.putText(frame, name, (bbox[0], bbox[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(imgBackground, reg_id,
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            current_date = datetime.datetime.now().date()
            if student_id and morn_attendance:
                t = threading.Thread(target=morningattendance, args=(
                    name, current_date, roll_no, div, branch, reg_id))
                t.start()
            if student_id and even_attendance:
                t = threading.Thread(
                    target=eveningattendance, args=(name, current_date))
                t.start()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route of video feed to flask webpage on index page
@app.route('/video1')
def video1():
    try:
        camera1 = params['camera_index_1']
        camera = cv2.VideoCapture(camera1)
        return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print("Error:", e)
        return "Error connecting to the video stream"


@app.route('/video2')
def video2():
    try:
        camera2 = params['camera_index_2']
        camera = cv2.VideoCapture(camera2)
        return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print("Error:", e)
        return "Error connecting to the video stream"


# Route which displays the attendance of all student for that current day
@app.route('/display_attendance', methods=['GET', 'POST'])
@login_required
def display_attendance():
    if current_user.role == 'student':
        stop_camera()
        current_date = datetime.datetime.now().date()
        try:
            input_date = None
            if request.method == 'POST':
                input_date = request.form['date']
            if input_date is None:
                date = current_date
            else:
                date = input_date
            data = Attendance.query.filter_by(date=date).all()
            return render_template('display_data.html', data=data, date=date)
        except Exception as e:
            # Return a more informative error message or handle specific exceptions
            return str(e)
    else:
        return 'UnAuthorized access'

# Route to add new students page for admins


@app.route('/data')
@login_required
def data():
    if current_user.role == 'admin':
        stop_camera()
        return render_template('data.html')
    else:
        return 'UnAuthorized Access'


@app.route('/add_user', methods=['POST'])
@login_required
def add_user():
    name = request.form['name']
    branch = request.form['branch']
    division = request.form['division']
    regid = request.form['reg_id']
    rollno = request.form['roll_no']

    # Check if a student with the same name already exists
    existing_student = Student_data.query.filter_by(name=name).first()

    if existing_student:
        # Student already exists, handle the error (e.g., display a message)
        error_message = 'Student already exists!'
        flash('Student already exists!', 'error')
        return redirect(url_for('data'))
    else:
        # Check if the post request has the file part
        if 'image' not in request.files:
            error_message = 'No file part'
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            error_message = 'No selected file'
            flash('No selected file')
            return redirect(request.url)

        # Check if the file extension is allowed
        if file and allowed_file(file.filename):
            # Secure the filename to prevent any malicious activity
            filename = secure_filename(
                regid + '.' + file.filename.rsplit('.', 1)[1].lower())
            # Save the file to the upload folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Proceed to add the new student
            user = Student_data(name=name, rollno=rollno,
                                division=division, branch=branch, regid=regid)
            db.session.add(user)
            db.session.commit()
            error_message = 'Student added successfully!'
            flash('Student added successfully!', 'success')
            return render_template('data.html', error=error_message)
        else:
            error_message = 'Invalid file extension. Allowed extensions are: png, jpg, jpeg, gif'
            flash(
                'Invalid file extension. Allowed extensions are: png, jpg, jpeg, gif', 'error')
            return redirect(request.url)


@app.route('/get_attendance', methods=['GET'])
@login_required
def get_attendance():
    if current_user.role == 'teacher':
        stop_camera()
        query_parameters = {}
        for key, value in request.args.items():
            if value:
                query_parameters[key] = value

        if query_parameters:
            attendance_records = Attendance.query.filter_by(
                **query_parameters).order_by(asc(Attendance.reg_id)).all()

            if not attendance_records:
                flash("No records available for the specified criteria")
        else:
            flash("No parameters provided for query")
            attendance_records = []  # Assign an empty list to avoid undefined variable error

        return render_template('results.html', attendance_records=attendance_records)
    else:
        return 'UnAuthorized access'

# Function to download the attendance of particular date in cvs format


@app.route('/download_attendance_csv', methods=['POST'])
def download_attendance_csv():
    try:
        # Assuming the date is submitted via a form
        date = request.form.get('date')
        if not date:
            flash("Date not provided for downloading.")
            return redirect(url_for('get_attendance'))

        # Retrieve attendance records for the specified date
        attendance_records = Attendance.query.filter_by(date=date).all()

        if not attendance_records:
            flash("No attendance records found for the specified date.")
            return redirect(url_for('get_attendance'))

        # Create a CSV string
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Name', 'Start Time', 'End Time', 'Date',
                        'Roll Number', 'Division', 'Branch', 'Registration ID'])
        for record in attendance_records:
            writer.writerow([record.name, record.start_time, record.end_time, record.date,
                            record.roll_no, record.division, record.branch, record.reg_id])

        # Save CSV file to a specified folder
        folder_path = 'downloads'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, f"attendance_records_{date}.csv")
        with open(file_path, 'w') as f:
            # Remove trailing newline characters
            f.write(output.getvalue().strip())

        flash("Attendance records downloaded successfully.")
        error_message = 'Attendance records downloaded successfully.'
        return render_template('results.html', error=error_message)
    except Exception as e:
        logging.exception(
            "Error occurred while generating CSV file: %s", str(e))
        flash("An error occurred while generating CSV file.")
        error_message = 'An error occurred while generating CSV file.'
        return render_template('results.html', error=error_message)


# Route to registration page for viewing the attendance
@app.route('/register', methods=['GET', 'POST'])
def register():
    stop_camera()
    error = None  # Initialize error variable
    password_regex = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    if request.method == 'POST':
        username = request.form['username']
        reg_id = request.form['reg_id']
        password = request.form['password']
        role = request.form['role']
        hashed_pass = bcrypt.generate_password_hash(password).decode('utf-8')
        # Check if username or reg_id already exists
        existing_user = Users.query.filter_by(username=username).first()
        existing_reg_id = Users.query.filter_by(reg_id=reg_id).first()

        if existing_user:
            error = 'Username already exists!'
            print('Username already exists!')
        elif existing_reg_id:
            error = 'Registration ID already exists!'
            print('Registration ID already exists!')
        elif not re.match(password_regex, password):
            error = 'Password must contain at least one uppercase letter, one symbol, one number, and be at least 8 characters long!'
        else:
            # Create new user
            new_user = Users(username=username, reg_id=reg_id,
                             password=hashed_pass, role=role)
            db.session.add(new_user)
            db.session.commit()
            error = 'Registration successfull!'
            flash('Registration successful!', 'success')
            return render_template('login.html', error=error)

    # Pass error variable to template
    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        try:
            user = Users.query.filter(Users.username == username).first()

            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user)
                session['user_id'] = user.id
                session['username'] = user.username
                session['role'] = user.role
                error_message = 'Welcome back, {}!'.format(user.username)
                flash(error_message, 'success')
                # Redirect based on the user's role
                if user.role == 'admin':
                    flash(error_message, 'success')
                    return render_template('data.html', error=error_message)
                elif user.role == 'teacher':
                    flash(error_message, 'success')
                    return render_template('results.html', error=error_message)
                elif user.role == 'student':
                    flash(error_message, 'success')
                    return render_template('display_data.html', error=error_message)
            else:
                error_message = 'Incorrect username or password. Please try again.',
                flash('Incorrect username or password. Please try again.', 'error')
        except SQLAlchemyError as e:
            error_message = 'An error occurred while processing your request. Please try again later.'
            flash(
                'An error occurred while processing your request. Please try again later.', 'error')
            # Log the exception for further investigation
            print(e)
    # If the request method is not GET or POST, or if the login process fails for any reason
    return render_template('login.html', error=error_message)


def findEncodings(imageslist):
    encodeList = []
    for img in imageslist:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Route to trigger encoding manually


@app.route('/generate_encodings', methods=['GET', 'POST'])
def generate_encodings():
    if request.method == 'POST':
        # Delete existing encoding file if it exists
        encoding_file_path = "Resources/EncodeFile.p"
        if os.path.exists(encoding_file_path):
            os.remove(encoding_file_path)
            print("File removed")
            flash("File Removed")

        # Importing the student images
        folderPath = 'uploads'
        pathList = os.listdir(folderPath)
        imgList = []
        studentIds = []
        for path in pathList:
            imgList.append(cv2.imread(os.path.join(folderPath, path)))
            studentIds.append(os.path.splitext(path)[0])
            print(os.path.splitext(path)[0])
        # Generate encodings
        try:
            print("Encoding started...")
            error_message = 'Encoding started...'
            flash("Encoding started...", "success")
            encodeListKnown = findEncodings(imgList)
            encodeListKnownWithIds = [encodeListKnown, studentIds]
            print("Encoding complete")
            error_message = 'Encoding complete'
            flash("Encoding complete", "success")
            with open(encoding_file_path, 'wb') as file:
                pickle.dump(encodeListKnownWithIds, file)
            print("File Saved")
            error_message = 'Encodings generated successfully!'
            flash('Encodings generated successfully!', 'success')
        except Exception as e:
            print("Error:", e)
            flash('Error occurred while generating encodings.', 'error')

        # Redirect to homepage or any other page after encoding
        return redirect(url_for('data'))

    return render_template('data.html', error=error_message)


# Function for logout functionality
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    error_message = 'Logout Successfully!!'
    logout_user()
    session.clear()
    return render_template('login.html', error=error_message)


@app.route('/images')
@login_required
def images():
    if current_user.role == 'admin':
        image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(
            os.path.join(app.config['UPLOAD_FOLDER'], f))]
        image_no = len(image_files)
        print(f"No of images: {image_no}")
        return render_template('image_gallery.html', image_files=image_files, image_no=image_no)
    else:
        return 'UnAuthourized access'


@app.route('/images/<filename>')
def get_image(filename):
    # Serve a specific image file
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if current_user.role == 'student':
        name = session['username']
        data = Attendance.query.filter_by(name=name).all()
        no_of_attendance = len(data)
        return render_template('profile.html', data=data, no_of_attendance=no_of_attendance)


# Route to the index page where the camera feed is displayed
@app.route('/')
def index():
    start_camera()
    return render_template('index.html')


# Function to start to the app
if __name__ == '__main__':
    # app.run(debug=True,ssl_context=("cert.pem", "key.pem"))
    # app.run(debug=True)
    hostedapp.run(debug=True, ssl_context=(
        cert_path, key_path), host='0.0.0.0')
