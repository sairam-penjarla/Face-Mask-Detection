import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from config import IMAGE_SIZE

# Load the trained model
model = load_model('face_mask_detection_model.h5')

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_predict_mask(frame, model):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    faces_list = []
    preds = []
    
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, IMAGE_SIZE)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        faces_list.append(face)
        
        if len(faces_list) > 0:
            preds = model.predict(faces_list)
    
    return faces, preds

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    faces, preds = detect_and_predict_mask(frame, model)
    
    for (face, pred) in zip(faces, preds):
        (x, y, w, h) = face
        (mask, withoutMask) = pred[0]
        
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Display the resulting frame
    cv2.imshow('Webcam Face Mask Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
