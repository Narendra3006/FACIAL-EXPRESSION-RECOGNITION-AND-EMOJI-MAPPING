import cv2
import torch
import numpy as np
from model import EmotionCNN

# Emoji mapping
emoji_paths = {
    0: "angry.jpg",
    1: "disgust.jpg",
    2: "fear.jpg",
    3: "happy.jpg",
    4: "sad.jpg",
    5: "surprise.jpg",
    6: "neutral.jpg"
}

emotion_labels = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

def load_model(model_path):
    model = EmotionCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_tensor = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return face_tensor

# NEW: returns annotated frame with emoji
def predict_emotion(frame, model, face_cascade_path="haarcascade_frontalface_default.xml"):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Load emojis (cached to global dict)
    if not hasattr(predict_emotion, "emojis"):
        predict_emotion.emojis = {}
        for emotion, path in emoji_paths.items():
            emoji = cv2.imread(path)
            if emoji is None:
                raise FileNotFoundError(f"Emoji not found: {path}")
            predict_emotion.emojis[emotion] = cv2.resize(emoji, (100, 100))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emoji_x, emoji_y = 10, 10  # top-left corner for emoji

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_tensor = preprocess_face(face_roi)

        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output.data, 1)
            emotion = predicted.item()

        # Overlay emoji
        emoji = predict_emotion.emojis[emotion]
        frame[emoji_y:emoji_y+100, emoji_x:emoji_x+100] = emoji

        # Add label text
        label = emotion_labels[emotion]
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = emoji_x + (100 - text_size[0]) // 2
        text_y = emoji_y + 130  # below emoji

        # Background for text
        cv2.rectangle(frame, 
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5),
                      (0, 0, 0), -1)

        # Text label
        cv2.putText(frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        break  # Detect only first face for performance

    return frame

