from inference import video_enojify, load_model

def main():
    # Load trained model
    model = load_model("emotion_model.pth")
    # Run real-time inference
    video_enojify(model)

if __name__ == "__main__":
    main()