from dataset_scanner import DatasetScanner
from pathlib import Path

DATASET_ROOT = Path(__file__).parent.parent.parent / "data_sorted"
scanner = DatasetScanner(DATASET_ROOT)
train, val = scanner.scan()
scanner.save(train, val)
