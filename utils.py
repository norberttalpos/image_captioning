import torch
import torchvision.transforms as transforms

from dataset import Vocabulary

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def save_checkpoint(state, filename="model.pt"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint"),
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    return epoch


def load_model(checkpoint, model):
    print("=> Loading model")
    model.load_state_dict(checkpoint["model_state_dict"])


def create_vocabulary(*args):
    vocabulary = Vocabulary()
    vocabulary.build_vocabulary(*args)

    return vocabulary
