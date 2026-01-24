import torch

def sequence_collate(batch):
    """
    batch: list of (seq, length, label)
    """
    sequences, lengths, labels = zip(*batch)

    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    max_len = max(lengths)
    D = sequences[0].shape[1]

    padded = torch.zeros(len(sequences), max_len, D)

    for i, seq in enumerate(sequences):
        padded[i, : seq.shape[0]] = seq

    return padded, lengths, labels
