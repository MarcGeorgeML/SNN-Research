import sys
import argparse
from pathlib import Path
import torch

# Add necessary paths to load components
root = Path(__file__).parent.parent
sys.path.append(str(root / "Train"))
sys.path.append(str(root))

from inference.preprocessor import InferencePreprocessor
from Model.SpikEmo_Model import SpikEmo
from Model.spikformer import Spikformer
from spikingjelly.activation_based import functional

def load_snn_model(checkpoint_path, device="cuda"):
    print(f"Loading SpikEmo model from {checkpoint_path}...")
    
    # Model configuration from Train/train_spikemo.py config
    spikformer_model = Spikformer(
        depths=2,
        T=8,
        tau=10.0,
        common_thr=1.0,
        dim=256,
        heads=8
    )

    model = SpikEmo(
        dataset="custom",
        multi_attn_flag=True,
        roberta_dim=768,
        hidden_dim=1024,
        dropout=0,
        num_layers=6,
        model_dim=256,
        num_heads=4,
        D_m_audio=512,
        D_m_visual=1000,
        n_classes=6,
        spikformer_model=spikformer_model
    )
    
    # Load weights if path is provided and exists
    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Using untrained weights for demonstration.")
        
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="SpikEmo Single Video Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to the .mp4 video file")
    # parser.add_argument("--checkpoint", type=str, default="", help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu)")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file {video_path} not found.")
        return
        
    print("--- Step 1: Initialization ---")
    preprocessor = InferencePreprocessor(device=args.device)
    
    print("\n--- Step 2: Feature Extraction ---")
    features = preprocessor.process_video(video_path)
    
    if features[0] is None:
        print("Failed to extract features. Exiting.")
        return
        
    texts, audios, visuals = features
    
    # Add batch dimension (B=1, seq_len=num_segments, feat_dim)
    # The SpikEmo model expects (B, seq_len, feat_dim). 
    texts = texts.unsqueeze(0).to(args.device)
    audios = audios.unsqueeze(0).to(args.device)
    visuals = visuals.unsqueeze(0).to(args.device)
    
    num_segments = texts.shape[1]
    print(f"Extracted {num_segments} valid segments.")
    print(f"Text features shape: {texts.shape}")
    print(f"Audio features shape: {audios.shape}")
    print(f"Visual features shape: {visuals.shape}")
    
    checkpoint = "checkpoints/checkpoint_epoch44_f10.7124.pt"
    print("\n--- Step 3: SNN Model Inference ---")
    model = load_snn_model(checkpoint, device=args.device)
    
    with torch.no_grad():
        # The SpikEmo model returns: f_t, f_a, f_v, fusion_features, logits
        _, _, _, _, logits = model(texts, audios, visuals)
        
        # SNN models require calling reset_net after each forward pass / batch
        functional.reset_net(model)
        
    print(f"\nRaw Logits: {logits}")
    
    # Simple argmax for prediction
    probs = torch.softmax(logits, dim=-1)
    pred_class = torch.argmax(logits, dim=-1).item()
    confidence = probs[0][pred_class].item()
    
    # Output presentation
    print("\n" + "="*40)
    print(f"Prediction result for {video_path.name}:")
    print(f"Predicted Class ID : {pred_class}")
    print(f"Confidence         : {confidence:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
