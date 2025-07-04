import os
from torch.utils.data import Dataset

class PitchVideoDataset(Dataset):
    def __init__(self, root_dir, classes):
        """
        Args:
            root_dir (str): Root directory containing class subfolders with videos.
            classes (list): List of pitch class folder names (e.g., ['fastball', 'curveball', 'changeup'])
        """
        self.root_dir = root_dir
        self.classes = classes

        self.samples = []  # list of (video_path, label_index)
        for idx, cls in enumerate(classes):
            class_dir = os.path.join(root_dir, cls)
            for file in os.listdir(class_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(class_dir, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        return video_path, label
