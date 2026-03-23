from pathlib import Path
import pickle
from phase2_features.build_features import FeatureBuilder

TRAIN_SEGMENTS = Path("preprocessing/segments/train_segments.pkl")
VAL_SEGMENTS = Path("preprocessing/segments/val_segments.pkl")

TEST_SAMPLES = 50


def load_subset(path, n):
    with open(path, "rb") as f:
        segments = pickle.load(f)
    return segments[:n]


def main():

    print("Loading small subset for testing...")

    train_segments = load_subset(TRAIN_SEGMENTS, TEST_SAMPLES)
    val_segments = load_subset(VAL_SEGMENTS, TEST_SAMPLES)
    builder = FeatureBuilder(output_root="features_test")
    print("Running Phase-2 TEST pipeline")
    builder.run(train_segments, val_segments)
    print("\nTest pipeline completed.")
    print("Output saved to:")
    print("features_test/")


if __name__ == "__main__":
    main()

# python preprocessing/phase2_features/test.py
