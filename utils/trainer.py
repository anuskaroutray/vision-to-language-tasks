import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import compute_individual_metrics
from utils.loss import CaptioningLoss

def train_captioning_model(model, train_dataloader, validation_dataloader, args, optimizer = None, device = "cpu",
			temperature = 0.7):

	model.train()
	# NOTE: Set criterion
	captioning_criterion = CaptioningLoss()
	image_concepts_criterion = nn.BCEWithLogitsLoss()

	if not optimizer:
		optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)

	for epoch in range(args.max_epochs):
		total_loss = 0
		with tqdm(train_dataloader, unit = "batch", position = 0, leave = True) as tepoch:
			for i, batch in enumerate(tepoch):
				tepoch.set_description(f"Epoch {epoch}")

				concept_vector, all_probability_distributions = model.forward(batch["image"].to(device), batch["word_embeddings"].to(device))
				loss = captioning_criterion(all_probability_distributions) + \
						image_concepts_criterion(concept_vector, batch["ground_concept_vector"].squeeze().to(device))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				total_loss += loss.item()
				tepoch.set_postfix(loss = total_loss / (i+1))

		evaluate_captioning_model(model, train_dataloader, args, device = device, temperature = temperature)
		evaluate_captioning_model(model, validation_dataloader, args, device = device, temperature = temperature)
	
	# TODO: Write code for logging

def evaluate_captioning_model(model, dataloader, args, device = "cpu", temperature = 0.7):

	captioning_criterion = CaptioningLoss()
	image_concepts_criterion = nn.BCEWithLogitsLoss()
	model.eval()

	total_loss = 0
	with tqdm(dataloader, unit = "batch", position = 0, leave = True) as tepoch:
		for i, batch in enumerate(tepoch):
			tepoch.set_description(f"Evaluating")

			concept_vector, all_probability_distributions = model.forward(batch["image"].to(device), batch["word_embeddings"].to(device))
			loss = captioning_criterion(all_probability_distributions) + \
					image_concepts_criterion(concept_vector, batch["ground_concept_vector"].squeeze().to(device))

			total_loss += loss.item()
			tepoch.set_postfix(loss = total_loss / (i+1))

def evaluate_captioning_metrics(true_captions, predicted_captions):

	all_metrics = {}

	for true_caption, predicted_caption in zip(true_captions, predicted_captions):
		metrics_dict = compute_individual_metrics(true_captions, predicted_captions)
		for key in metrics_dict.keys():
			if key not in all_metrics:
				all_metrics[key] = 0
			all_metrics[key] += metrics_dict[key]

	for key in all_metrics.keys():
		all_metrics[key] /= len(true_captions)

	print("The evaluation metrics are as follows: \n")
	for key, val in all_metrics.items():
		print(f"{key}:\t{val}")