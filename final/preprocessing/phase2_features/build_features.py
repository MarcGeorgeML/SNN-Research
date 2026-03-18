import pickle
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from video_decoder import extract_audio, extract_frames
from text_features import TextFeatureExtractor
from audio_features import AudioFeatureExtractor
from visual_features import VisualFeatureExtractor


class FeatureBuilder:

    def __init__(self, output_root="features", device="cuda", batch_size=32):

        self.output_root = Path(output_root)
        self.output_root.mkdir(exist_ok=True)

        self.batch_size = batch_size

        self.text_encoder = TextFeatureExtractor(device=device)
        self.audio_encoder = AudioFeatureExtractor(device=device)
        self.visual_encoder = VisualFeatureExtractor(device=device)

    def load_segments(self, file_path):

        with open(file_path, "rb") as f:
            return pickle.load(f)

    def group_by_video(self, segments):

        groups = defaultdict(list)

        for s in segments:
            groups[s["video_path"]].append(s)

        return groups

    def build_split(self, segments, split_name):

        output_dir = self.output_root / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        text_features = []
        audio_features = []
        visual_features = []
        labels = []

        video_groups = self.group_by_video(segments)

        for video_path, video_segments in tqdm(video_groups.items()):

            try:
                audio = extract_audio(video_path)
            except Exception as e:
                print(f"Audio decode failed: {video_path}")
                print(e)
                continue

            batch_texts = []
            batch_audio = []
            batch_frames = []
            batch_labels = []

            for segment in video_segments:

                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                label = segment["label"]

                try:

                    frames = extract_frames(video_path, start, end)

                    if len(frames) == 0:
                        continue

                    audio_segment = self.audio_encoder.slice_segment(audio, start, end)

                    if len(audio_segment) == 0:
                        continue

                    if text.strip() == "":
                        continue

                    batch_texts.append(text)
                    batch_audio.append(audio_segment)
                    batch_frames.append(frames)
                    batch_labels.append(label)

                except Exception as e:
                    print(f"Segment failed: {video_path}")
                    print(e)

                if len(batch_texts) == self.batch_size:

                    self.process_batch(
                        batch_texts,
                        batch_audio,
                        batch_frames,
                        batch_labels,
                        text_features,
                        audio_features,
                        visual_features,
                        labels,
                    )

                    batch_texts, batch_audio, batch_frames, batch_labels = [], [], [], []

            if batch_texts:

                self.process_batch(
                    batch_texts,
                    batch_audio,
                    batch_frames,
                    batch_labels,
                    text_features,
                    audio_features,
                    visual_features,
                    labels,
                )

        if len(text_features) == 0:
            raise RuntimeError("No features extracted. Check earlier pipeline errors.")

        text_tensor = torch.stack(text_features)
        audio_tensor = torch.stack(audio_features)
        visual_tensor = torch.stack(visual_features)
        label_tensor = torch.tensor(labels)

        with open(output_dir / "text.pkl", "wb") as f:
            pickle.dump({"features": text_tensor, "labels": label_tensor}, f)

        with open(output_dir / "audio.pkl", "wb") as f:
            pickle.dump({"features": audio_tensor, "labels": label_tensor}, f)

        with open(output_dir / "visual.pkl", "wb") as f:
            pickle.dump({"features": visual_tensor, "labels": label_tensor}, f)

    def process_batch(
        self,
        texts,
        audios,
        frames_list,
        batch_labels,
        text_store,
        audio_store,
        visual_store,
        label_store,
    ):


        text_emb = self.text_encoder.encode(texts)
        audio_emb = self.audio_encoder.encode(audios)
        all_frames = []
        frame_counts = []

        for frames in frames_list:
            frame_counts.append(len(frames))

            for frame in frames:
                all_frames.append(self.visual_encoder.transform(frame))

        if len(all_frames) == 0:
            return

        frame_tensor = torch.stack(all_frames).to(self.visual_encoder.device)

        with torch.no_grad():
            features = self.visual_encoder.cnn(frame_tensor)
            features = features.flatten(1)
            
        idx = 0
        
        for i, count in enumerate(frame_counts):
            segment_features = features[idx:idx + count]
            idx += count
            mean_feat = segment_features.mean(dim=0)
            max_feat = segment_features.max(dim=0).values
            pooled = 0.5 * mean_feat + 0.5 * max_feat
            visual_emb = self.visual_encoder.projection(pooled)
            text_store.append(text_emb[i])
            audio_store.append(audio_emb[i])
            visual_store.append(visual_emb.cpu())
            label_store.append(batch_labels[i])

    def run(self, train_segments, val_segments):

        print("\nBuilding TRAIN features")
        self.build_split(train_segments, "train")

        print("\nBuilding VALIDATION features")
        self.build_split(val_segments, "validation")

        print("\nFeature extraction finished.")