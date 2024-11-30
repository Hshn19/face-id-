from flask import Flask, render_template, request, redirect, url_for, session
import face_recognition
import cv2
import numpy as np
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key for production

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    encoding = db.Column(db.PickleType, nullable=False) # Store face encoding

# Create the database
with app.app_context():
    db.create_all()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        # Start capturing video from the webcam
        video_capture = cv2.VideoCapture(0)

        print("Please look at the camera for face registration...")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert the image from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Draw rectangles around detected faces and store the encoding for registration
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Check if we have a valid encoding and register it
                if len(face_encodings) > 0:
                    # Save user data to the database
                    new_user = User(username=username, password=password, encoding=encoding.tobytes())
                    db.session.add(new_user)
                    db.session.commit()
                    cv2.putText(frame, "Face Registered!", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    break

            # Display the resulting frame
            cv2.imshow('Register Your Face', frame)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close windows
        video_capture.release()
        cv2.destroyAllWindows()

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['username'] = username  # Store username in session
            return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return f'Welcome {session["username"]}!'
    return redirect(url_for('login'))


@app.route('/auto_login', methods=['GET'])
def auto_login():
    # Start capturing video from the webcam for automatic login
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for unknown_encoding in face_encodings:
            for user in User.query.all():
                stored_encoding = np.frombuffer(user.encoding)
                # Convert bytes back to array
                match = face_recognition.compare_faces([stored_encoding], unknown_encoding)
                if match[0]: # If a match is found
                    session["username"] = user.username
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return redirect(url_for('dashboard'))

        # Display the resulting frame with detected faces highlighted
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow('Auto-Login - Look at Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        return "No matching face found."

    if __name__ == '__main__':
        app.run(debug=True)

