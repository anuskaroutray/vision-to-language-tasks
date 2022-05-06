import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from utils.loss import ContrastiveLoss

def train(model, train_dataloader, validation_dataloader, args, optimizer = None, device = "cpu"):

	# NOTE: Set criterion
	criterion = None

	if not optimizer:
		optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)

	for epoch in range(args.max_epochs):
		total_loss = 0
		with tqdm(train_dataloader, unit = "batch", position = 0, leave = True) as tepoch:
			for i, batch in enumerate(tepoch):
				tepoch.set_description(f"Epoch {epoch}")
		# for i, batch in enumerate(tqdm(train_dataloader, position = 0, leave = True)):
				# image_features, text_features = model.forward(batch['images'].to(device), batch['caption_input_ids'].to(device), 
				# 										batch['caption_attention_masks'].to(device), batch['caption_token_type_ids'].to(device))

				# loss = criterion(image_features, text_features, batch['image_ids'].to(device))
				# optimizer.zero_grad()
				# loss.backward()
				# optimizer.step()

				# total_loss += loss.item()
				# tepoch.set_postfix(loss=total_loss/(i+1))

				model.forward(batch["image"].to(device), batch["word_embeddings"].to(device))
				exit(0)
	
	# TODO: Write code for logging