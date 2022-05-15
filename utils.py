import torch
import torchvision.transforms as transforms

from dataset import Vocabulary

# normalizing transform (values based on ImageNet)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def save_checkpoint(state, filename="model.pt"):
    """
    Saves a model checkpoint
    :param state: current state of model
    :param filename: filename to save
    :return:
    """

    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """
    Load model and optimizer from saved checkpoint
    :param checkpoint: saved model checkpoint
    :param model: model
    :param optimizer: optimizer
    :return: number of trained epochs of saved model
    """
    print("=> Loading checkpoint"),
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    return epoch


def load_model(checkpoint, model):
    """
    Load model from saved checkpoint
    :param checkpoint: saved model checkpoint
    :param model: model
    :return:
    """

    print("=> Loading model")
    model.load_state_dict(checkpoint["model_state_dict"])


def create_vocabulary(*args):
    """
    Create vocabulary from caption files
    :param args: caption files
    :return: created vocabulary
    """
    vocabulary = Vocabulary()
    vocabulary.build_vocabulary(*args)

    return vocabulary
