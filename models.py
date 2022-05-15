import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained cnn
        inception = models.inception_v3(pretrained=True, aux_logits=False)
        modules = list(inception.children())[:-2]
        self.cnn = nn.Sequential(*modules)

        # converting the feature vector to the same dimension as the embedding
        # TODO: adaptive pooling
        self.adaptive_pool = nn.AdaptiveMaxPool2d(14)

    def forward(self, images):
        out = self.cnn(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)

        return out


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim, attention_dim):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)

        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size, bias=True)

        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, features, captions, caption_lengths):
        caption_lengths = torch.tensor(caption_lengths)

        batch_size = features.size(0)
        encoder_dim = features.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = features.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = captions.permute(1, 0)[sort_ind]

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths) - 1, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths) - 1, num_pixels).to(device)

        for t in range(max(decode_lengths) - 1):
            batch_size_t = sum([l > t for l in decode_lengths])

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.lstm(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            preds = self.linear(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, caption_lengths, alphas, sort_ind


class ImageCaptioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim, attention_dim):
        super().__init__()
        self.encoder = Encoder()

        self.vocab_size = vocab_size

        self.decoder = Decoder(embed_size, hidden_size, vocab_size, encoder_dim, attention_dim)

    def forward(self, images, captions, caption_lengths):
        features = self.encoder(images)

        return self.decoder(features, captions, caption_lengths)

    def caption_image(self, image, vocabulary, max_length=30):

        sentence = []
        with torch.no_grad():
            # Encode
            encoder_out = self.encoder(image)
            encoder_dim = encoder_out.size(3)

            # Flatten encoding
            encoder_out = encoder_out.view(1, -1, encoder_dim)

            prev_word = torch.LongTensor([vocabulary.stoi['<SOS>']]).to(device)

            step = 1

            h, c = self.decoder.init_hidden_state(encoder_out)

            while step < max_length:

                embeddings = self.decoder.embedding(prev_word).squeeze(1)
                attention_weighted_encoding, _ = self.decoder.attention(encoder_out, h)
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding

                h, c = self.decoder.lstm(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))

                scores = self.decoder.linear(h)
                scores = F.log_softmax(scores, dim=1)
                best = scores.argmax(1)[0]

                sentence.append(best.item())

                prev_word = best.unsqueeze(0)

                if vocabulary.itos[best.item()] == "<EOS>":
                    break

                step += 1

        return [vocabulary.itos[idx] for idx in sentence]
