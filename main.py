from dataset import get_data_loaders
from train import train
from inference import video_emojify, load_model

def main():
    train_loader, val_loader, emotions = get_data_loaders()

    model = train(train_loader, val_loader, epochs=100, lr=0.0001)

    model = load_model("emojify_model.pth")

    video_emojify(model, emotions)


if __name__ == "__main__":
    main()