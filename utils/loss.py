# Import required libraries
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_metric_learning import losses

# TODO: Write the captioning loss and image concept predictor loss