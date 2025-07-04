import cv2
import torch
import numpy as np
from torchvision import transforms
from vision.hsv_filter import HSV_Filter, MIN_AREA, MAX_AREA  # your exact filter and constants

# Define transforms for CNN input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # adjust if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video {video_path} has no frames")

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)

    frame_dict = {}
    current_frame = 0
    retrieved = 0
    while cap.isOpened() and retrieved < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame == indices[retrieved]:
            frame_dict[current_frame] = frame
            retrieved += 1
        current_frame += 1

    cap.release()

    frames = [frame_dict.get(i, list(frame_dict.values())[-1]) for i in indices]
    return frames

def apply_hsv_filter(frames, hsv_filter=None):
    if hsv_filter is None:
        hsv_filter = HSV_Filter(h_min=0, h_max=179, s_min=0, s_max=50, v_min=200, v_max=255)

    filtered_frames = []
    for frame in frames:
        filtered = hsv_filter.apply(frame)
        filtered_frames.append(filtered)
    return filtered_frames

def preprocess_frames(frames):
    processed = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = transform(frame_rgb)
        processed.append(processed_frame)
    return processed

def process_video(video_path, num_frames=16, hsv_filter=None):
    frames = load_video_frames(video_path, num_frames)
    filtered_frames = apply_hsv_filter(frames, hsv_filter)
    processed_frames = preprocess_frames(filtered_frames)
    frames_tensor = torch.stack(processed_frames)  # Shape: (num_frames, 3, 224, 224)
    return frames_tensor
