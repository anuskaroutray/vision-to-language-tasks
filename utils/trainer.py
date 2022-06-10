import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import compute_individual_metrics
from utils.loss import CaptioningLoss, VQALoss

# Function to train image captioning model
def train_captioning_model(model, train_dataloader, validation_dataloader, args, optimizer = None, device = "cpu",
			temperature = 0.7):

	model.train()
	captioning_criterion = CaptioningLoss()
	image_concepts_criterion = nn.BCEWithLogitsLoss()

	# Create the optimizer
	if not optimizer:
		optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)

	# Train for multiple number of epochs
	for epoch in range(args.max_epochs):
		total_loss = 0
		with tqdm(train_dataloader, unit = "batch", position = 0, leave = True) as tepoch:
			for i, batch in enumerate(tepoch):
				tepoch.set_description(f"Epoch {epoch}")

				concept_vector, all_probability_distributions = model.forward(batch["image"].to(device), batch["word_embeddings"].to(device))
				
				# Loss is composed of captioning loss and image concepts loss
				loss = captioning_criterion(all_probability_distributions) + \
						image_concepts_criterion(concept_vector, batch["ground_concept_vector"].squeeze().to(device))

				# Backpropagation step
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				total_loss += loss.item()
				tepoch.set_postfix(loss = total_loss / (i+1))

		# Evaluate the model on training and validation data after every epoch
		evaluate_captioning_model(model, train_dataloader, args, device = device, temperature = temperature)
		evaluate_captioning_model(model, validation_dataloader, args, device = device, temperature = temperature)

# Function to evaluate the captioning model
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

# Function to compute evaluation metrics on captioning data
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

# Function to train VQA model
def train_vqa_model(model, train_dataloader, validation_dataloader, args, optimizer = None, device = "cpu",
			temperature = 0.7):

	model.train()
	vqa_criterion = VQALoss()
	image_concepts_criterion = nn.BCEWithLogitsLoss()

	if not optimizer:
		optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)

	for epoch in range(args.max_epochs):
		total_loss = 0
		total = 0
		correct = 0
		with tqdm(train_dataloader, unit = "batch", position = 0, leave = True) as tepoch:
			for i, batch in enumerate(tepoch):
				tepoch.set_description(f"Epoch {epoch}")

				concept_vector, answer_distribution = model.forward(batch["image"].to(device), batch["word_embeddings"].to(device),
																	batch["question_embeddings"].to(device))
				
				# Loss is composed of VQA loss and image concepts loss
				loss = vqa_criterion(answer_distribution, batch["answer"].to(device)) + \
						image_concepts_criterion(concept_vector, batch["ground_concept_vector"].squeeze().to(device))

				# Backpropagation step
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				total_loss += loss.item()
				tepoch.set_postfix(loss = total_loss / (i+1))

				# Compute the accuracy of the Visual Question Answering model
				_, predicted = torch.max(answer_distribution, dim = 1)
				total += batch["answer"].size(0)

				correct += (predicted == batch["answer"].to(device)).sum().item()
				tepoch.set_postfix(acc = correct / total)

		# Evaluate the VQA model after every epoch
		evaluate_vqa_model(model, train_dataloader, args, device = device, temperature = temperature)
		evaluate_vqa_model(model, validation_dataloader, args, device = device, temperature = temperature)

# Function to evaluate VQA model
def evaluate_vqa_model(model, dataloader, args, device = "cpu", temperature = 0.7):

	model.eval()
	vqa_criterion = VQALoss()
	image_concepts_criterion = nn.BCEWithLogitsLoss()

	all_ground_answers = []
	all_predicted_answers = []

	total_loss = 0
	total = 0
	correct = 0
	with tqdm(dataloader, unit = "batch", position = 0, leave = True) as tepoch:
		for i, batch in enumerate(tepoch):
			tepoch.set_description(f"Evaluating")

			concept_vector, answer_distribution = model.forward(batch["image"].to(device), batch["word_embeddings"].to(device),
																batch["question_embeddings"].to(device))
			loss = vqa_criterion(answer_distribution, batch["answer"].to(device)) + \
					image_concepts_criterion(concept_vector, batch["ground_concept_vector"].squeeze().to(device))

			total_loss += loss.item()
			tepoch.set_postfix(loss = total_loss / (i+1))

			_, predicted = torch.max(answer_distribution, dim = 1)
			total += batch["answer"].size(0)

			correct += (predicted == batch["answer"].to(device)).sum().item()
			tepoch.set_postfix(acc = correct / total)