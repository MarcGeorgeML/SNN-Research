import pickle
from pathlib import Path
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):

    def __init__(self, root_dir):

        root_dir = Path(root_dir)
        with open(root_dir / "text.pkl",   "rb") as f: text_data   = pickle.load(f)
        with open(root_dir / "audio.pkl",  "rb") as f: audio_data  = pickle.load(f)
        with open(root_dir / "visual.pkl", "rb") as f: visual_data = pickle.load(f)
        self.text = text_data["features"]
        self.audio = audio_data["features"]
        self.visual = visual_data["features"]
        self.labels = text_data["labels"]

        assert len(self.text) == len(self.audio) == len(self.visual)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "visual": self.visual[idx],
            "label": self.labels[idx],
        }
