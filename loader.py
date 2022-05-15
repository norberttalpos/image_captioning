from dataset import CaptionDataset, CollateFN
from torch.utils.data import DataLoader


def get_loader(
        root_folder,
        annotation_file,
        transform,
        vocab,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    dataset = CaptionDataset(root_folder, annotation_file, transform=transform, vocab=vocab)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CollateFN(pad_idx=pad_idx),
    )

    return loader, dataset
