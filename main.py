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
		device = torch.device("cuda:1")
		
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

	model = model.to(device)

	train(model, train_dataloader, validation_dataloader, args, device = device)