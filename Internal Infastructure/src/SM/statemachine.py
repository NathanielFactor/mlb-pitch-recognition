import torch
from torchvision import transforms

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import cnn_model
from models import lstm_model
from vision import hsv_filter
from vision import video_utils
from data import data_loader


class PitchRecognitionSM:
    states = ['idle', 'video_received', 'preprocessing', 'feature_extraction', 'classification', 'result_output', 'error_handling']

    def __init__(self):
        # Data attributes
        self.video_path = None
        self.frames = None
        self.filtered_frames = None
        self.cnn_features = None
        self.prediction = None

        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn = cnn_model.CNNFeatureExtractor().to(self.device)
        self.lstm = lstm_model.PitchClassifierLSTM().to(self.device)

        # Load pretrained weights after training
        # Uncomment the following lines if you have saved model weights
        # self.cnn.load_state_dict(torch.load('cnn.pth'))
        # self.lstm.load_state_dict(torch.load('lstm.pth'))
        self.cnn.eval()
        self.lstm.eval()

    def validate_video(self):
        # Example: check if video exists and readable
        if self.video_path is None or not os.path.isfile(self.video_path):
            print(f"Error: video file not found at {self.video_path}")
            self.fail()
            return
        print("Video validated successfully")
        self.preprocess()

    def extract_frames_and_filter(self):
        # Use video_utils to extract frames, then apply hsv_filter on each frame
        try:
            self.frames = video_utils.extract_frames(self.video_path, num_frames=16)
            self.filtered_frames = [hsv_filter.apply(frame) for frame in self.frames]
            print(f"Extracted and filtered {len(self.filtered_frames)} frames")
            self.extract_features() 
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            self.fail()

    def run_cnn(self):
        try:
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            # Prepare batch tensor
            batch = torch.stack([preprocess(frame) for frame in self.filtered_frames]).to(self.device)
            # Pass through CNN
            with torch.no_grad():
                features = self.cnn(batch)  # (batch_size, output_dim)
            # Reshape for LSTM input: (1, seq_len, feature_dim)
            self.cnn_features = features.unsqueeze(0)
            print("CNN feature extraction complete")
            self.classify()
        except Exception as e:
            print(f"Error in CNN feature extraction: {e}")
            self.fail()

    def run_lstm(self):
        try:
            with torch.no_grad():
                logits = self.lstm(self.cnn_features)  # (1, num_classes)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            self.prediction = predicted_class
            print(f"Pitch classified as class {self.prediction}")
            self.output_result()
        except Exception as e:
            print(f"Error in LSTM classification: {e}")
            self.fail()

    def send_result(self):
        # Example: print or return prediction
        print(f"Predicted pitch type: {self.prediction}")
        # Reset state machine for next run
        self.reset()

    def handle_error(self):
        print("An error occurred. Resetting state machine.")
        self.reset()


if __name__ == "__main__":

    sm = PitchRecognitionSM()

    try:
        done = False
        while not done:
            sm.validate_video()
            sm.extract_frames_and_filter()
            sm.run_cnn()
            sm.run_lstm()
            sm.send_result()
            done = True

    except KeyboardInterrupt:
        sm.handle_error()
