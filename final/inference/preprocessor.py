import sys
from pathlib import Path
import torch

# Add preprocessing modules to path
root = Path(__file__).parent.parent
sys.path.append(str(root / "preprocessing" / "phase1_segmentation"))
sys.path.append(str(root / "preprocessing" / "phase2_features"))

from whisper_segmenter import WhisperSegmenter
from video_decoder import extract_audio, extract_frames
from text_features import TextFeatureExtractor
from audio_features import AudioFeatureExtractor
from visual_features import VisualFeatureExtractor

class InferencePreprocessor:
    """
    End-to-end preprocessing pipeline for a single .mp4 file.
    Initializes all feature extractors once and keeps them in memory
    for fast inference.
    """
    def __init__(self, device="cuda"):
        self.device = device
        
        print("Loading Whisper Segmenter...")
        self.segmenter = WhisperSegmenter(device=device)
        
        print("Loading Text Encoder...")
        self.text_encoder = TextFeatureExtractor(device=device)
        
        print("Loading Audio Encoder...")
        self.audio_encoder = AudioFeatureExtractor(device=device)
        
        print("Loading Visual Encoder...")
        self.visual_encoder = VisualFeatureExtractor(device=device)
        
        print("All encoders loaded successfully.")

    def process_video(self, video_path):
        """
        Processes a single .mp4 file and returns its multimodal features.
        
        Args:
            video_path (str or Path): Path to the input video.
            
        Returns:
            tuple: (text_tensor, audio_tensor, visual_tensor) representing
                   the extracted features for the video's segments.
        """
        video_path = str(video_path)
        print(f"Segmenting audio for {video_path}...")
        
        # 1. Segment video
        segments = self.segmenter.segment_audio(video_path)
        
        if not segments:
            print("No speech segments found.")
            return None, None, None
            
        print(f"Found {len(segments)} segments. Extracting audio...")
        
        # 2. Extract full audio
        audio = extract_audio(video_path)
        
        text_features = []
        audio_features = []
        visual_features = []
        
        print("Extracting multi-modal features per segment...")
        
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            
            # Extract frames for this segment
            frames = extract_frames(video_path, start, end)
            if len(frames) == 0:
                continue
                
            # Slice audio
            audio_segment = self.audio_encoder.slice_segment(audio, start, end)
            if len(audio_segment) == 0:
                continue
                
            if text.strip() == "":
                continue
                
            # Process text and audio
            t_emb = self.text_encoder.encode([text])[0]
            a_emb = self.audio_encoder.encode([audio_segment])[0]
            
            # Process visual
            frame_tensors = [self.visual_encoder.transform(f) for f in frames]
            frame_tensor = torch.stack(frame_tensors).to(self.device)
            
            with torch.no_grad():
                v_feats = self.visual_encoder.cnn(frame_tensor)
                v_feats = v_feats.flatten(1)
                
            mean_feat = v_feats.mean(dim=0)
            max_feat = v_feats.max(dim=0).values
            pooled = 0.5 * mean_feat + 0.5 * max_feat
            v_emb = self.visual_encoder.projection(pooled)
            
            text_features.append(t_emb)
            audio_features.append(a_emb)
            visual_features.append(v_emb.cpu())
            
        if len(text_features) == 0:
            print("No valid features could be extracted from segments.")
            return None, None, None
            
        print("Preprocessing complete!")
        return (
            torch.stack(text_features),
            torch.stack(audio_features),
            torch.stack(visual_features)
        )
