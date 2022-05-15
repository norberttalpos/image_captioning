from os import listdir
from os import path
from os.path import isfile, join

import torch
import torchvision.transforms as transforms
from PIL import Image

from models import ImageCaptioner
from params import *
from utils import load_model
from utils import normalize, create_vocabulary


def to_sentence(tokens):
    return " ".join(tokens[1:-1])


def print_examples(model, device, vocab):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    model.eval()

    test_examples_path = "test_examples"
    file_names = [f for f in listdir(test_examples_path) if isfile(join(test_examples_path, f))]

    test_imgs = [transform(Image.open(join(test_examples_path, f)).convert("RGB")).unsqueeze(0) for f in file_names]

    for (img, file_name) in zip(test_imgs, file_names):
        print(file_name + ": " + to_sentence(model.caption_image(img.to(device), vocab))[:-1])

    model.train()


def evaluate():
    vocab = create_vocabulary(captions_train_file, captions_val_file)
    vocab_size = len(vocab)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageCaptioner(embed_size, hidden_size, vocab_size, encoder_dim, attention_dim)
    model.to(device)

    if path.exists("model.pt"):
        load_model(torch.load("model.pt"), model)
    else:
        raise RuntimeError()

    print_examples(model, device, vocab)


if __name__ == "__main__":
    evaluate()
