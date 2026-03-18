from multiprocessing import freeze_support
from build_dataloader import build_dataloaders


def main():
    train_loader, _ = build_dataloaders("features", batch_size=16, num_workers=8)

    for batch in train_loader:
        texts, audios, visuals, speaker_mask, utterance_mask, labels = batch
        print("texts:", texts.shape)
        print("audios:", audios.shape)
        print("visuals:", visuals.shape)
        print("speaker_mask:", speaker_mask.shape)
        print("utterance_mask:", utterance_mask.shape)
        print("labels:", labels.shape)
        break


if __name__ == "__main__":
    freeze_support()
    main()
