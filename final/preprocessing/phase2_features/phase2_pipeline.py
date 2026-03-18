from pathlib import Path
import time

from build_features import FeatureBuilder


TRAIN_SEGMENTS = Path("segments/train_segments.pkl")
VAL_SEGMENTS = Path("segments/val_segments.pkl")


def main():

    if not TRAIN_SEGMENTS.exists():
        raise FileNotFoundError(f"Missing {TRAIN_SEGMENTS}")

    if not VAL_SEGMENTS.exists():
        raise FileNotFoundError(f"Missing {VAL_SEGMENTS}")

    print("Initializing FeatureBuilder")

    builder = FeatureBuilder(
        output_root="features",
        device="cuda",
        batch_size=32
    )

    print("Loading segment files")

    train_segments = builder.load_segments(TRAIN_SEGMENTS)
    val_segments = builder.load_segments(VAL_SEGMENTS)

    print(f"Train segments: {len(train_segments)}")
    print(f"Validation segments: {len(val_segments)}")

    start = time.time()

    builder.run(train_segments, val_segments)

    end = time.time()

    duration = end - start

    print("\nPhase-2 pipeline finished")
    print(f"Total runtime: {duration/60:.2f} minutes")


if __name__ == "__main__":
    main()

# python preprocessing/phase2_features/phase2_pipeline.py
