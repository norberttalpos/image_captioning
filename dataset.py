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


class Vocabulary:
    def __init__(self, min_frequency=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.min_frequency = min_frequency

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, *args):
        frequencies = Counter()
        idx = len(self.stoi)

        sentence_list = []

        for arg in args:
            df = pd.read_csv(arg, delimiter=csv_sep_regexp, keep_default_na=False)
            sentence_list.extend(df[caption_column].tolist())

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies.update([word])

                if frequencies.get(word) == self.min_frequency:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class CaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir

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
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        caplen = len(numericalized_caption)

        return img, torch.tensor(numericalized_caption), torch.tensor(caplen)


class CollateFN:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        caplens = [item[2] for item in batch]

        return imgs, targets, caplens
