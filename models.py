import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder Network
    Responsible for mapping the images into latent space
    """

    def __init__(self):
        """
        Initializes the Encoder network
        """

        super().__init__()

        # pretrained inception cnn
        inception = models.inception_v3(pretrained=True, aux_logits=False)

        # remove last two layers (used for classification)
        modules = list(inception.children())[:-2]
        self.cnn = nn.Sequential(*modules)

        # create output of dimension (14, 14, 2048)
        self.adaptive_pool = nn.AdaptiveMaxPool2d(14)

    def forward(self, images):
        """
        Forward step
        Maps the input image into latent feature space
        :param images: batch of images to encode
        :return: encoded images
        """

        out = self.cnn(images)
        out = self.adaptive_pool(out)

        # convert to rgb format
        out = out.permute(0, 2, 3, 1)

        return out


class Attention(nn.Module):
    """
    Attention network
    Responsible for creating weights for pixels based on the relevance to the current captioning step
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Initializes the Attention network
        :param encoder_dim: output dim of encoder
        :param decoder_dim: output dim of decoder
        :param attention_dim: attention dim
        """

        super().__init__()

        # creates attention weights based encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)

        # creates attention weights based on decoder (lstm) output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)

        # create attention weights from previous two
        self.full_att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward step
        Calculates attention weights for pixels
        :param encoder_out: encoder output
        :param decoder_hidden: hidden state of decoder
        :return:
        """

        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)

        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)

        # attention weights (sum == 1)
        alpha = self.softmax(att)

        # attention weighted image
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    """
    Decoder network
    Responsible for decoding the encoder output into a sequence (the caption)
    """
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim, attention_dim):
        """
        Initializes the Decoder network
        :param embed_dim: token embedding dimension
        :param decoder_dim: lstm cell hidden state dimension
        :param vocab_size: size of vocabulary
        :param encoder_dim: encoder dim
        :param attention_dim: attention dim
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size

        # create floats from long indices
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # attention network
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # lstm cell
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # layers to create lstm state from encoder output
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # layer to create a sigmoid activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.sigmoid = nn.Sigmoid()

        # layer to map the decoder output to values of vocab_size dimension
        self.linear = nn.Linear(decoder_dim, vocab_size)

        self.dropout = nn.Dropout(0.5)

    def init_hidden_state(self, encoder_out):
        """
        Init state of lstm cell
        :param encoder_out: encoder output
        :return: hidden state (h: hidden, c: cell)
        """

        # mean through image_size dimension
        mean_encoder_out = encoder_out.mean(dim=1)

        # create hidden state
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(self, features, captions, caption_lengths):
        """
        Forward step
        Creates the caption sequence from latent feature space (using attention weights at each step)
        :param features: encoder output (latent features of images)
        :param captions: captions of images
        :param caption_lengths: lengths of captions
        :return: predictions, encoded_captions, caption_lengths, alphas (attention weights), sort_ind (sorting indices)
        """

        caption_lengths = torch.tensor(caption_lengths)

        batch_size = features.size(0)
        encoder_dim = features.size(-1)

        # flatten image
        encoder_out = features.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # sort by lengths (to only process the batch entries with longer captions then current step (teacher forcing))
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = captions.permute(1, 0)[sort_ind]

        # embed captions into latent space
        embeddings = self.embedding(encoded_captions)

        # init lstm state
        h, c = self.init_hidden_state(encoder_out)

        # remove "end" token (we want the model to predict it)
        decode_lengths = (caption_lengths - 1).tolist()

        # store prediction and attention weights through the lstm process
        predictions = torch.zeros(batch_size, max(decode_lengths) - 1, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths) - 1, num_pixels).to(device)

        # while all target captions are processed (teacher forcing)
        # create attention weighted image encoding based on previous token, feed it as input (with previous token)
        # generate lstm cell output
        # save
        for t in range(max(decode_lengths) - 1):
            # num of entries to process
            batch_size_t = sum([l > t for l in decode_lengths])

            # [:batch_size_t], because the tensors are sorted with respect to caption length
            # create attention weights
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            # lstm forget gate
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # feed into lstm
            h, c = self.lstm(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            # map hidden state to predictions
            preds = self.linear(self.dropout(h))

            # save predictions and attention weights
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, caption_lengths, alphas, sort_ind


class ImageCaptioner(nn.Module):
    """
    Image Captioner network
    Responsible for handling the encoder-decoder networks
    """

    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim, attention_dim):
        """
        Initializes the Image Captioner network
        :param embed_dim: embedding dim
        :param decoder_dim: decoder dim
        :param vocab_size: vocabulary size
        :param encoder_dim: encoder dim
        :param attention_dim: attention dim
        """

        super().__init__()

        self.encoder = Encoder()
        self.vocab_size = vocab_size
        self.decoder = Decoder(embed_dim, decoder_dim, vocab_size, encoder_dim, attention_dim)

    def forward(self, images, captions, caption_lengths):
        """
        Forward step
        Propagates the images through the encoder, then the encoder output and captions through the decoder
        :param images: images
        :param captions: captions
        :param caption_lengths: captions_lengths
        :return: decoder network output
        """

        features = self.encoder(images)

        return self.decoder(features, captions, caption_lengths)

    def caption_image(self, image, vocabulary, max_length=30):
        """
        Responsible for inference
        Creates caption for an image
        :param image: image to create the caption for
        :param vocabulary: vocabulary
        :param max_length: max length of caption
        :return: the caption (in tokenized format)
        """

        sentence = []
        with torch.no_grad():
            # encode image
            encoder_out = self.encoder(image)
            encoder_dim = encoder_out.size(3)

            # flatten encoding
            encoder_out = encoder_out.view(1, -1, encoder_dim)

            prev_word = torch.LongTensor([vocabulary.stoi['<SOS>']]).to(device)

            step = 1

            # init lstm state
            h, c = self.decoder.init_hidden_state(encoder_out)

            while step < max_length:

                # embed captions into latent space
                embeddings = self.decoder.embedding(prev_word).squeeze(1)

                # create attention weights
                attention_weighted_encoding, _ = self.decoder.attention(encoder_out, h)

                # lstm forget gate
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding

                # feed into lstm
                h, c = self.decoder.lstm(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))

                # map hidden state to predictions
                scores = self.decoder.linear(h)

                # softmax to create probabilites for tokens
                scores = F.log_softmax(scores, dim=1)

                # choose the idx with the highest probability
                best = scores.argmax(1)[0]

                sentence.append(best.item())

                prev_word = best.unsqueeze(0)

                # if the model predicted the "end" token, break
                if vocabulary.itos[best.item()] == "<EOS>":
                    break

                step += 1

        # map indices to tokens
        return [vocabulary.itos[idx] for idx in sentence]
