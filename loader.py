from dataset import CaptionDataset, CollateFN
from torch.utils.data import DataLoader


def get_loader(
        img_folder,
        captions_file,
        transform,
        vocab,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    """

    :param img_folder: image folder
    :param captions_file: caption file
    :param transform: pre-process transforms
    :param vocab: vocabulary
    :param batch_size: batch_size
    :param num_workers: number of workers while loading
    :param shuffle: whether to shuffle the dataset
    :param pin_memory: pin_memory
    :return:
    """
    dataset = CaptionDataset(img_folder, captions_file, transform=transform, vocab=vocab)

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
