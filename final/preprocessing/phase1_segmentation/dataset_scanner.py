import random
from pathlib import Path

class DatasetScanner:
    def __init__(self, dataset_root, train_split=0.8, seed=42):
        self.dataset_root = Path(dataset_root)
        self.train_split = train_split
        self.seed = seed

        self.label_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5
        }

        random.seed(self.seed)

    def scan(self):
        train_samples = []
        val_samples = []

        for emotion, label in self.label_map.items():
            video_dir = self.dataset_root / emotion / "video"

            if not video_dir.exists():
                raise FileNotFoundError(f"Missing directory: {video_dir}")

            videos = sorted(video_dir.glob("*.mp4"))

            if len(videos) == 0:
                print(f"Warning: no videos found in {video_dir}")
                continue

            videos = [(str(v), label) for v in videos]
            random.shuffle(videos)
            split_idx = int(len(videos) * self.train_split)
            train_samples.extend(videos[:split_idx])
            val_samples.extend(videos[split_idx:])
        return train_samples, val_samples

    def save(self, train_samples, val_samples, output_dir="."):

        # Save purely into the phase1_segmentation folder itself
        save_dir = Path(__file__).parent
        train_file = save_dir / "train_videos.txt"
        val_file = save_dir / "val_videos.txt"

        with open(train_file, "w") as f:
            for path, label in train_samples:
                f.write(f"{path} {label}\n")

        with open(val_file, "w") as f:
            for path, label in val_samples:
                f.write(f"{path} {label}\n")

        print(f"Saved {len(train_samples)} training videos to {train_file}")
        print(f"Saved {len(val_samples)} validation videos to {val_file}")
