import torch


def multimodal_collate(batch):

    texts = torch.stack([x["text"] for x in batch]).detach()
    audios = torch.stack([x["audio"] for x in batch]).detach()
    visuals = torch.stack([x["visual"] for x in batch]).detach()
    labels = torch.tensor([x["label"] for x in batch])
    batch_size = texts.shape[0]
    texts = texts.unsqueeze(0)
    audios = audios.unsqueeze(0)
    visuals = visuals.unsqueeze(0)
    speaker_mask = torch.ones(1, batch_size, 1)
    utterance_mask = torch.ones(batch_size, 1)
    padded_labels = labels.unsqueeze(1)
    return (texts, audios, visuals, speaker_mask, utterance_mask, padded_labels)
