# Import the required libraries
import torch
import timm, time
import torchvision
import torchmetrics
import torch.nn as nn
from utils.env import *
from torch import optim
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Type, Any, Callable, Union, List, Optional

# TODO:
#   1. Code for Image Captioning

class ConceptAndLocalFeaturePredictor(nn.Module):

    def __init__(self, model_name, hidden_size = 2048, output_size = 512, pretrained = False):
        super(ConceptAndLocalFeaturePredictor, self).__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretrained = pretrained

        model = timm.create_model(self.model_name, pretrained = self.pretrained)
        hidden_layer = nn.Linear(model.num_features, self.hidden_size)
        concept_layer = nn.Linear(self.hidden_size, self.output_size)

        model.reset_classifier(0)
        self.model = nn.Sequential(OrderedDict([
            ('backbone', model),
            ('hidden_layer', hidden_layer),
            ('output_layer', concept_layer)
        ]))

    def concept_forward(self, x):
        x = self.model(x)
        return x
    
    def local_feature_forward(self, x):
        x = self.model.backbone.features(x)
        return x
        
class ImageCaptioningModel(nn.Module):

    def __init__(self, model_name, input_size = 49, hidden_size = 2048, output_size = 512, 
                pretrained = False, bidirectional = True, epsilon = 0.6, device = "cpu"):
        super(ImageCaptioningModel, self).__init__()

        self.model_name = model_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretrained = pretrained
        self.bidirectional = bidirectional
        self.epsilon = epsilon
        self.device = device
        
        self.concept_and_local_feature_predictor = ConceptAndLocalFeaturePredictor(self.model_name,
                    hidden_size = self.hidden_size, output_size = self.output_size, pretrained = self.pretrained)

        self.local_image_gru = nn.GRU(input_size = self.input_size, hidden_size = self.output_size, 
                                        bidirectional = self.bidirectional, batch_first = True)

        self.concept_transform_layer = nn.Linear(self.output_size, self.output_size, bias = True)
        self.visual_transform_layer = nn.Linear(self.output_size, self.output_size, bias = True)
        self.concept_transform_layer_transpose = nn.Linear(self.output_size, self.output_size, bias = True)
        self.visual_transform_layer_transpose = nn.Linear(self.output_size, self.output_size, bias = True)

        # TODO: Add the layers corresponding to image captioning task


    def forward(self, images, word_embeddings):

        concept_vector = self.concept_and_local_feature_predictor.concept_forward(images)
        concept_set = ((concept_vector > torch.tensor(self.epsilon)).float() * 1).unsqueeze(-1) * torch.eye(self.output_size).reshape((1, 
                            self.output_size, self.output_size)).repeat(concept_vector.size(0), 1, 1).to(self.device)

        # NOTE: Local feature is of shape (7, 7, 512), but in the paper, it is (14, 14, 512)
        # Will check back later how to obtain conv5_4 feature map without max pooling
        
        local_feature = self.concept_and_local_feature_predictor.local_feature_forward(images)
        local_feature = local_feature.view((local_feature.shape[0], local_feature.shape[1], -1))
        
        local_feature_out, local_feature_hidden = self.local_image_gru(local_feature)
        local_feature_out = local_feature_out.view(local_feature_out.size(0), local_feature.size(1), 2, self.output_size)
        local_feature_out = local_feature_out.reshape((2, local_feature_out.size(0), local_feature.size(1), -1))
        visual_representation = local_feature_out[0] + local_feature_out[1]

        final_image_rep = self.semantic_guided_attention(concept_set, visual_representation)

        # TODO: Write code for image captioning, i.e., word attention and sentence generation. 
        # NOTE: In this function, return concept vector for concept prediction loss

        



    def semantic_guided_attention(self, concept_set, visual_representation):
        
        visual_transform = self.visual_transform_layer(visual_representation)
        visual_transform_transpose = self.visual_transform_layer_transpose(visual_representation)

        concept_transform = self.concept_transform_layer(concept_set)
        concept_transform_transpose = self.concept_transform_layer_transpose(concept_set)

        visual_similarity = torch.bmm(visual_transform, concept_transform)
        concept_similarity = torch.bmm(visual_transform_transpose, concept_transform_transpose)

        visual_attention_weights = torch.exp(torch.max(visual_similarity, dim = 2).values) / torch.exp(torch.sum(visual_similarity, dim = 2))
        concept_based_image_rep = torch.sum(visual_representation * visual_attention_weights.unsqueeze(-1), dim = 2)

        concept_attention_weights = torch.exp(torch.max(concept_similarity, dim = 2).values) / torch.exp(torch.sum(concept_similarity, dim = 2))
        region_based_concept_rep = torch.sum(concept_set * concept_attention_weights.unsqueeze(-1), dim = 2)

        final_image_rep = torch.cat([concept_based_image_rep, region_based_concept_rep], dim = 1)
        return final_image_rep