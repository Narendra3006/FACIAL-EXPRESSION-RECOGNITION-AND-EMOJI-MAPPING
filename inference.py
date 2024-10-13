import numpy as np
import time
import cv2
import torch
from model import EmojifyModel
from utils import plot_image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_model(model_path='emojify_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmojifyModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def detect_and_emojify(img_path, model, emotions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=0)
        face_tensor = torch.tensor(face, dtype=torch.float32).to(device)

        with torch.no_grad():
            prediction = model(face_tensor)
            idx = torch.argmax(prediction, dim=1).item()

        emoj = cv2.imread(f'{emotions[idx]}.jpg')
        plot_image(img, emoj)

def video_emojify(model, emotions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to capture video")
            break
        
        img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 

                face = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=0)
                face_tensor = torch.tensor(face, dtype=torch.float32).to(device)

                with torch.no_grad():
                    prediction = model(face_tensor)
                    idx = torch.argmax(prediction, dim=1).item()

                emoj_path = f'{emotions[idx]}.jpg'
                emoj = cv2.imread(emoj_path)

                if emoj is None:
                    print(f"Error: Could not load emoji file {emoj_path}")
                    emoj = np.random.randn(150, 150) 
                else:
                    emoj = cv2.resize(emoj, (w, h))

                img[y:y+h, x:x+w] = cv2.addWeighted(img[y:y+h, x:x+w], 0.5, emoj, 0.5, 0)

        #else:
         #   emoj = cv2.imread('NofaceDetected.jpeg')
          #  emoj = cv2.resize(emoj, (100, 100)) 
           # img = cv2.addWeighted(img, 0.5, emoj, 0.5, 0)

        cv2.imshow('Video Emojify', img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


