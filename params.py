# hyperparams of model and teaching

embed_size = 512
hidden_size = 512
attention_dim = 512
encoder_dim = 2048
num_layers = 1
learning_rate = 3e-4
num_epochs = 100

# data

dataset = "8k"

dataset_folder = "flickr8k/images" if dataset == "8k" else "flickr30k_images/images"
images_folder = "flickr8k/images" if dataset == "8k" else "flickr30k_images/images"
captions_train_file = "flickr8k/captions_train.txt" if dataset == "8k" else "flickr30k_images/captions_train.txt"
captions_val_file = "flickr8k/captions_val.txt" if dataset == "8k" else "flickr30k_images/captions_val.txt"

image_column = "image" if dataset == "8k" else "image_name"
caption_column = "caption" if dataset == "8k" else "comment"

csv_sep_regexp = "," if dataset == "8k" else "\t"
