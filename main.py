# Import the required libraries

import os
import torch
import argparse
import torch.nn as nn
from utils.env import set_seed
# from utils.loss import ContrastiveLoss
from torch.utils.data import DataLoader
from data.dataloader import ImageCaptionDataset
# from utils.test import test_retrieval, TestCallback
from utils.trainer import train
# from src.model import LanguageModel, VisionModel, DualEncoder, SpatialInformationAggregatorModule
from src.model import ImageCaptioningModel

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default = 128, type = int)	# 64, 128
	parser.add_argument('--dataset', default = 'flickr8k', type = str, choices=['coco', 'flickr30k', 'flickr8k'])
	parser.add_argument('--max_epochs', default = 10, type = int)
	parser.add_argument('--dropout', default = 0.4, type = float)
	parser.add_argument('--eval', default = True, type = bool)
	parser.add_argument('--gradient_clip_val', default = 30.0, type = float)
	parser.add_argument('--gpus', default = "0,1", type = str) # 1 -> [1]; 0,1 -> [0, 1]
	parser.add_argument('--image_resize', default = (224, 224), type = tuple)
	parser.add_argument('--learning_rate', default = 1e-5, type = float)
	parser.add_argument('--max_length_caption', default = 64, type = int)
	parser.add_argument('--preprocess_text', default = True, type = bool) # With and without
	parser.add_argument('--seed', default = 0, type = int)
	parser.add_argument('--test', default = False, type = bool)
	parser.add_argument('--test_shuffle', default = False, type = bool)
	parser.add_argument('--train', default = True, type = bool)
	parser.add_argument('--train_shuffle', default = True, type = bool)
	parser.add_argument('--val_shuffle', default = False, type = bool)
	parser.add_argument('--validation', default = True, type = bool)
	parser.add_argument('--warmup_epochs', default = 2, type = int)
	parser.add_argument('--warn_grayscale', default = False, type = bool)
	parser.add_argument('--weight_decay', default = 1e-4, type = float)

	parser.add_argument('--model_name', default = "vgg19", type = str)
	parser.add_argument('--input_size', default = 49, type = int)
	parser.add_argument('--hidden_size', default = 2048, type = int)
	parser.add_argument('--output_size', default = 512, type = int)
	parser.add_argument('--pretrained', default = True, type = bool)
	parser.add_argument('--bidirectional', default = True, type = bool)
	parser.add_argument('--epsilon', default = 0.6, type = float)

	args = parser.parse_args()

	set_seed(args.seed)
	args.gpus = [int(item) for item in args.gpus.split(',')]

	device = torch.device("cpu")
	if torch.cuda.is_available(): 
		device = torch.device("cuda:0")

	train_dataloader, validation_dataloader, test_dataloader = None, None, None
	if args.train:
		train_data = ImageCaptionDataset(args.dataset, preprocess_text = args.preprocess_text, split = "train", max_length_caption = args.max_length_caption,
				image_resize = args.image_resize, warn_grayscale = args.warn_grayscale, eval = False)
		train_dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = args.train_shuffle, collate_fn = train_data.collater)

	if args.validation:
		validation_data = ImageCaptionDataset(args.dataset, preprocess_text = args.preprocess_text, split = "val", max_length_caption = args.max_length_caption,
				image_resize = args.image_resize, warn_grayscale = args.warn_grayscale, eval = args.eval)
		validation_dataloader = DataLoader(validation_data, batch_size = args.batch_size, shuffle = args.val_shuffle, collate_fn = validation_data.collater)
	
	if args.test:
		test_data = ImageCaptionDataset(args.dataset, preprocess_text = args.preprocess_text, split = "test", max_length_caption = args.max_length_caption,
				image_resize = args.image_resize, warn_grayscale = args.warn_grayscale, eval = args.eval)
		test_dataloader = DataLoader(test_data, batch_size = args.batch_size, shuffle = args.test_shuffle, collate_fn = test_data.collater)

	model = ImageCaptioningModel(args.model_name, input_size = args.input_size, hidden_size = args.hidden_size, output_size = args.output_size, 
								pretrained = args.pretrained, bidirectional = args.bidirectional, epsilon = args.epsilon, device = device)

	model.to(device)

	train(model, train_dataloader, validation_dataloader, args, device = device)






















# if __name__ == "__main__":

# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--accelerator', default = "dp", type = str)	# 64, 128
# 	parser.add_argument('--batch_size', default = 128, type = int)	# 64, 128
# 	parser.add_argument('--checkpoint', default = 'checkpoints/retrieval.ckpt', type = str)
# 	parser.add_argument('--dataset', default = 'coco', type = str, choices=['cc', 'coco', 'flickr30k'])
# 	parser.add_argument('--de_max_epochs', default = 10, type = int)
# 	parser.add_argument('--deterministic', default = True, type = bool)
# 	parser.add_argument('--dropout', default = 0.4, type = float)
# 	parser.add_argument('--eval', default = True, type = bool)
# 	parser.add_argument('--gradient_clip_val', default = 30.0, type = float)
# 	parser.add_argument('--gpus', type = str) # 1 -> [1]; 0,1 -> [0, 1]
# 	parser.add_argument('--height', default = 7, type = int)
# 	parser.add_argument('--image_resize', default = (224, 224), type = tuple)
# 	parser.add_argument('--language_input_size', default = 768, type = int)
# 	parser.add_argument('--language_model_name', default = 'bert', type = str)
# 	parser.add_argument('--language_learning_rate', default = 1e-5, type = float)
# 	parser.add_argument('--language_pretrained', default = True, type = bool)
# 	parser.add_argument('--local_files_only', default = True, type = bool)
# 	parser.add_argument('--load_cache', default = True, type = bool)
# 	parser.add_argument('--max_length_caption', default = 64, type = int)
# 	parser.add_argument('--num_channels', default = 2048, type = int)
# 	parser.add_argument('--num_workers', default = 2, type = int)
# 	parser.add_argument('--output_size', default = 512, type = int) # Tune this [512, 256, 1024]
# 	parser.add_argument('--preprocess_text', default = True, type = bool) # With and without
# 	parser.add_argument('--progress_bar_refresh_rate', default = 5, type = int)
# 	parser.add_argument('--rpn_post_nms_top_n_test', default = 50, type = int)
# 	parser.add_argument('--seed', default = 0, type = int)
# 	parser.add_argument('--siam_learning_rate', default = 1e-4, type = float)
# 	parser.add_argument('--siam_max_epochs', default = 10, type = int)
# 	parser.add_argument('--siam_pretrained', default = False, type = bool)
# 	parser.add_argument('--test', default = False, type = bool)
# 	parser.add_argument('--test_shuffle', default = False, type = bool)
# 	parser.add_argument('--train', default = True, type = bool)
# 	parser.add_argument('--train_de', default = False, type = bool)
# 	parser.add_argument('--train_shuffle', default = True, type = bool)
# 	parser.add_argument('--train_siam', default = False, type = bool)
# 	parser.add_argument('--trainable_backbone_layers', default = 0, type = int)
# 	parser.add_argument('--val_shuffle', default = False, type = bool)
# 	parser.add_argument('--validation', default = True, type = bool)
# 	parser.add_argument('--vision_hidden_size', default = 2048, type = int)
# 	parser.add_argument('--vision_learning_rate', default = 1e-4, type = float)
# 	parser.add_argument('--vision_model_name', default = 'resnet50', type = str)
# 	parser.add_argument('--vision_pretrained', default = False, type = bool)
# 	parser.add_argument('--warmup_epochs', default = 2, type = int)
# 	parser.add_argument('--warn_grayscale', default = False, type = bool)
# 	parser.add_argument('--weight_decay', default = 1e-4, type = float)
# 	parser.add_argument('--width', default = 7, type = int)
# 	parser.add_argument('--recall_ks', default = "1,5", type = str)
	
# 	args = parser.parse_args()

# 	set_seed(args.seed)

# 	args.gpus = [int(item) for item in args.gpus.split(',')]
# 	recall_ks = [int(r) for r in args.recall_ks.split(',')]

# 	device = torch.device("cpu")
# 	if torch.cuda.is_available(): 
# 		device = torch.device("cuda:0")


# 	# Dataloaders
# 	train_dataloader, validation_dataloader, test_dataloader = None, None, None
# 	if args.train:
# 		train_data = ImageCaptionDataset(args.dataset, language_model_name = args.language_model_name, preprocess_text = args.preprocess_text, 
# 								split='train', max_length_caption = args.max_length_caption, local_files_only = args.local_files_only, 
# 								image_resize = args.image_resize, warn_grayscale = args.warn_grayscale)
# 		train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.train_shuffle, collate_fn=train_data.collater)

# 	args.dataset = 'flickr30k'
# 	if args.validation:
# 		validation_data = ImageCaptionDataset(args.dataset, language_model_name = args.language_model_name, preprocess_text = args.preprocess_text, 
# 								split='val', max_length_caption = args.max_length_caption, local_files_only = args.local_files_only, 
# 								image_resize = args.image_resize, warn_grayscale = args.warn_grayscale, eval = args.eval)
# 		validation_dataloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle = args.val_shuffle, collate_fn = validation_data.collater)
	
# 	if args.test:
# 		test_data = ImageCaptionDataset(args.dataset, language_model_name = args.language_model_name, preprocess_text = args.preprocess_text, 
# 								split='test', max_length_caption = args.max_length_caption, local_files_only = args.local_files_only, 
# 								image_resize = args.image_resize, warn_grayscale = args.warn_grayscale, eval = args.eval)
# 		test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle = args.test_shuffle, collate_fn = test_data.collater)

# 	# Models  
# 	# if args.train_de:  
# 	model_de = DualEncoder(args.vision_model_name, args.language_model_name, language_input_size = args.language_input_size, 
# 				vision_hidden_size = args.vision_hidden_size, output_size = args.output_size, vision_learning_rate = args.vision_learning_rate,
# 				language_learning_rate = args.language_learning_rate, dropout = args.dropout, vision_pretrained = args.vision_pretrained, 
# 				language_pretrained = args.language_pretrained, weight_decay = args.weight_decay, warmup_epochs = args.warmup_epochs)
	
# 	model_de = nn.DataParallel(model_de)
# 	model_de.to(device)
# 	train(model_de, train_dataloader, validation_dataloader, args, device = device)

		# model_siam = None

	# Training
	# if args.train_de:
		# de_trainer = pl.Trainer(max_epochs = args.de_max_epochs,
		# 					progress_bar_refresh_rate = args.progress_bar_refresh_rate, gpus = args.gpus, gradient_clip_val=args.gradient_clip_val)
							# add deterministic in Trainer if cannot reproduce results

		# if args.train and args.validation: 
		# 	de_trainer.fit(model_de, train_dataloader, validation_dataloader)
		
		# if args.train: 
		# 	de_trainer.fit(model_de, train_dataloader)

	# if not args.train_de:
	# 	model_de = DualEncoder.load_from_checkpoint(args.checkpoint)

	# if args.train_siam:

	# 	if not args.train_de:
	# 		if os.path.exists(args.checkpoint):
	# 			model_de = DualEncoder.load_from_checkpoint(args.checkpoint)
	# 		else:
	# 			model_de = DualEncoder(args.vision_model_name, args.language_model_name, language_input_size = args.language_input_size, 
	# 				vision_hidden_size = args.vision_hidden_size, output_size = args.output_size, vision_learning_rate = args.vision_learning_rate,
	# 				language_learning_rate = args.language_learning_rate, dropout = args.dropout, vision_pretrained = args.vision_pretrained, 
	# 				language_pretrained = args.language_pretrained, weight_decay = args.weight_decay, warmup_epochs = args.warmup_epochs)
		
	# 	model_siam = SpatialInformationAggregatorModule(model_de, height = args.height, width = args.width, num_channels = args.num_channels, 
	# 				output_size = args.output_size, pretrained = args.siam_pretrained, learning_rate = args.siam_learning_rate, weight_decay = args.weight_decay,
	# 				trainable_backbone_layers = args.trainable_backbone_layers, rpn_post_nms_top_n_test = args.rpn_post_nms_top_n_test)

	# 	siam_trainer = pl.Trainer(max_epochs = args.siam_max_epochs,
	# 						progress_bar_refresh_rate = args.progress_bar_refresh_rate, gpus = args.gpus, 
	# 						gradient_clip_val=args.gradient_clip_val, strategy = args.accelerator)

	# 	if args.train and args.validation: 
	# 		siam_trainer.fit(model_siam, train_dataloader, validation_dataloader)
		
	# 	if args.train: 
	# 		siam_trainer.fit(model_siam, train_dataloader)

	# if not args.train_siam:
	# 	model_siam = SpatialInformationAggregatorModule.load_from_checkpoint(args.checkpoint)

	# if not (args.train_de and args.train_siam):
	# 	model_de = DualEncoder.load_from_checkpoint(args.checkpoint)
	# 	model_siam = None
	
	# # Testing
	# if not args.test:
	# 	print("Evaluating on validation set")
	# 	if model_siam:
	# 		recalls = test_retrieval(model_siam, validation_dataloader, device, recall_ks, train_siam = True)
	# 	else:
	# 		recalls = test_retrieval(model_de, validation_dataloader, device, recall_ks, train_siam = False)

	# 	for i, k in enumerate(recall_ks):
	# 		print(f"Recall@{k}: ", recalls[i])
	# else:
	# 	print("Evaluating on test set")
	# 	if model_siam:
	# 		recalls = test_retrieval(model_siam, test_dataloader, device, recall_ks, train_siam = args.train_siam)
	# 	else:
	# 		recalls = test_retrieval(model_de, test_dataloader, device, recall_ks, train_siam = args.train_siam)
	# 	for i, k in enumerate(recall_ks):
	# 		print(f"Recall@{k}: ", recalls[i])