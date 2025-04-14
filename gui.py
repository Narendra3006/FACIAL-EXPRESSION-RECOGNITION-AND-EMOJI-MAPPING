import tkinter as tk
from tkinter import messagebox
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
from model import EmotionCNN

# Model & emoji config
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

class EmojiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Recognition and Emoji Mapping")
        self.root.geometry("1000x700")
        self.root.configure(bg='white')

        # Title
        tk.Label(root, text="FACIAL EXPRESSION RECOGNITION AND EMOJI MAPPING",
                 font=("Arial", 20, "bold"), bg='white', fg='black').pack(pady=10)

        # Video frame
        self.video_label = tk.Label(root, bg='white')
        self.video_label.pack(pady=10)

        # Start and Stop buttons
        self.btn_frame = tk.Frame(root, bg='white')
        self.btn_frame.pack(side=tk.BOTTOM, pady=20)

        self.start_btn = tk.Button(self.btn_frame, text="Start", bg="green", fg="white", font=("Arial", 14),
                                   command=self.start_camera, width=10)
        self.start_btn.pack(side=tk.LEFT, padx=20)

        self.stop_btn = tk.Button(self.btn_frame, text="Stop", bg="red", fg="white", font=("Arial", 14),
                                  command=self.stop_camera, width=10)
        self.stop_btn.pack(side=tk.RIGHT, padx=20)

        # Initialize vars
        self.cap = None
        self.running = False
        self.model = self.load_model("emotion_model.pth")  # Update model path if needed
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Load emojis
        self.emojis = {
            k: cv2.resize(cv2.imread(v), (100, 100))
            for k, v in emoji_paths.items()
        }

    def load_model(self, model_path):
        model = EmotionCNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def preprocess_face(self, face_img):
        face_img = cv2.resize(face_img, (48, 48))
        face_tensor = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        return face_tensor

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_tensor = self.preprocess_face(face_roi)

                    with torch.no_grad():
                        output = self.model(face_tensor)
                        _, predicted = torch.max(output.data, 1)
                        emotion = predicted.item()

                    label = emotion_labels[emotion]
                    emoji = self.emojis[emotion]

                    # Add emoji on frame
                    frame[10:110, 10:110] = emoji

                    # Add label
                    cv2.putText(frame, label, (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                # Convert and display frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmojiApp(root)
    root.mainloop()



