import pickle
import torch


def check_file(path, expected_dim):

    data = pickle.load(open(path, "rb"))

    features = data["features"]
    labels = data["labels"]

    print(f"\nChecking: {path}")
    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)

    assert not torch.isnan(features).any(), "NaNs detected in features"
    assert features.shape[1] == expected_dim, "Feature dimension mismatch"
    assert len(features) == len(labels), "Feature/label size mismatch"

    print("✓ File OK")


check_file("features/train/text.pkl", 768)
check_file("features/train/audio.pkl", 768)
check_file("features/train/visual.pkl", 2048)

check_file("features/validation/text.pkl", 768)
check_file("features/validation/audio.pkl", 768)
check_file("features/validation/visual.pkl", 2048)

print("\nAll checks passed.")
