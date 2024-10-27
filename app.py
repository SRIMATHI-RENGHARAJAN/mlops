from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
camera = cv2.VideoCapture(1)

# Define compliments for emotions
compliments = {
    "happy": "Keep smiling! You're doing great!",
    "sad": "It's okay to feel sad sometimes. Better days are coming!",
    "angry": "Take a deep breath! You can handle this!",
    "surprised": "Wow! What an interesting reaction!",
    "disgust": "Stay positive! You got this!",
    "fear": "Don't worry, you're stronger than you think!",
    "neutral": "Just be yourself! You're amazing!"
}

def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotion = "neutral"  # Default emotion
    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame, emotion

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, emotion = detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/compliment/<emotion>')
def compliment(emotion):
    return {"compliment": compliments.get(emotion, "You're doing great!")}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

