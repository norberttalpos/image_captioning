import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter

from params import image_column, caption_column, dataset, csv_sep_regexp

spacy_eng = spacy.load("en_core_web_sm")


def tokenizer_eng(text):
    """
    Tokenizes text
    :param text: the text to tokenize
    :return:
    """
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]


class Vocabulary:
    def __init__(self, min_frequency=5):
        """
        Initialize vocabulary with basic tokens
        :param min_frequency: frequency threshold for words in vocabulary
        """

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.min_frequency = min_frequency

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, *args):
        """
        Creates the vocabulary from the words found in caption files
        :param args: the caption files
        :return:
        """

        frequencies = Counter()
        idx = len(self.stoi)

        sentence_list = []

        for arg in args:
            df = pd.read_csv(arg, delimiter=csv_sep_regexp, keep_default_na=False)
            sentence_list.extend(df[caption_column].tolist())

        for sentence in sentence_list:
            for word in tokenizer_eng(sentence):
                # set frequency of word
                frequencies.update([word])

                # if above threshold, put into vocabulary with unique index
                if frequencies.get(word) == self.min_frequency:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        """
        Map to vocavulary indices from words
        :param text: the text to numericalize
        :return: indices in vocabulary corresponding to the original text
        """

        tokenized_text = tokenizer_eng(text)

        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class CaptionDataset(Dataset):
    def __init__(self, img_dir, captions_file, vocab, transform=None):
        """
        Initialize CaptionDataset
        :param img_dir: image folder path
        :param captions_file: captions file path
        :param vocab: vocabulary
        :param transform: pre-process transformations
        """

        self.img_dir = img_dir

        if dataset == "8k":
            df = pd.read_csv(captions_file)
        else:
            df = pd.read_csv(captions_file, sep=csv_sep_regexp, keep_default_na=False)

        self.transform = transform

        self.imgs = df[image_column]
        self.captions = df[caption_column]

        self.vocab = vocab

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Return a dataset entry at index
        :param index: index of entry
        :return: entry at index
        """

        # loading the captions and image
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.img_dir, img_id)).convert("RGB")

        # pre-processing the image
        if self.transform is not None:
            img = self.transform(img)

        # adding "start" and "end" tokens to the caption
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        # length of the caption
        caplen = len(numericalized_caption)

        return img, torch.tensor(numericalized_caption), torch.tensor(caplen)


class CollateFN:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        Return a batch of entries from dataset
        :param batch: idx of batch
        :return: fields of entries batched
        """
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]

        # pad shorter captions to be equal size
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        caplens = [item[2] for item in batch]

        return imgs, targets, caplens
