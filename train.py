from os import path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

from loader import get_loader
from models import ImageCaptioner
from params import *
from utils import normalize, create_vocabulary
from utils import save_checkpoint, load_checkpoint


def propagate_forward(device, model, criterion, imgs, captions, caplens):
    imgs = imgs.to(device)
    captions = captions.to(device)

    predictions, captions_sorted, caption_lengths, alphas, sort_ind = model(imgs, captions[:-1], caplens)

    targets = captions_sorted[:, 1:]

    loss = criterion(predictions.reshape(-1, predictions.shape[2]), targets.reshape(-1))

    # doubly stochastic regularization
    # encourages to sum the alphas at each pixel to 1 through the lstm timesteps
    loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

    return loss


def train_epoch(train_loader, device, model, criterion, optimizer):
    losses = []

    model.train()

    for idx, (imgs, captions, caplens) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        loss = propagate_forward(device, model, criterion, imgs, captions, caplens)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()

    mean_loss = np.mean(losses)

    return mean_loss


def val_epoch(val_loader, device, model, criterion):
    losses = []

    with torch.no_grad():
        model.eval()

        for idx, (imgs, captions, caplens) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            loss = propagate_forward(device, model, criterion, imgs, captions, caplens)
            losses.append(loss.item())

    mean_loss = np.mean(losses)

    return mean_loss


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    vocab = create_vocabulary(captions_train_file, captions_val_file)

    train_loader, train_dataset = get_loader(
        root_folder=dataset_folder,
        annotation_file=captions_train_file,
        transform=transform,
        vocab=vocab,
        num_workers=2,
    )

    val_loader, val_dataset = get_loader(
        root_folder=dataset_folder,
        annotation_file=captions_val_file,
        transform=transform,
        vocab=vocab,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    load_model = True
    save_model = True

    vocab_size = len(vocab)

    epoch = 0

    # initialize model, loss etc
    model = ImageCaptioner(embed_size, hidden_size, vocab_size, encoder_dim, attention_dim)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for name, param in model.encoder.cnn.named_parameters():
        param.requires_grad = "fc.weight" in name or "fc.bias" in name

    if load_model and path.exists("model.pt"):
        epoch = load_checkpoint(torch.load("model.pt"), model, optimizer)

    model.train()

    best_loss = 100000000
    epochs_since_improvement = 0

    while True:
        print("epoch: " + str(epoch))

        val_loss = train_epoch(train_loader, device, model, criterion, optimizer)
        # val_loss = val_epoch(val_loader, device, model, criterion)

        print("val loss: " + str(val_loss))

        is_best = val_loss < best_loss
        best_loss = max(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

            if save_model:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                }
                save_checkpoint(checkpoint)

        if epochs_since_improvement > 5:
            break

        print("")
        epoch += 1


if __name__ == "__main__":
    train()
