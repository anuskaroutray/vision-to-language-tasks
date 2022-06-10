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
                pretrained = False, bidirectional = True, epsilon = 0.6, device = "cpu",
                vocab_size = 7000, word_hidden_size = 512):
        super(ImageCaptioningModel, self).__init__()

        self.model_name = model_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretrained = pretrained
        self.bidirectional = bidirectional
        self.epsilon = epsilon
        self.device = device

        self.vocab_size = vocab_size
        self.word_hidden_size = word_hidden_size

        self.concept_and_local_feature_predictor = ConceptAndLocalFeaturePredictor(self.model_name,
                    hidden_size = self.hidden_size, output_size = self.output_size, pretrained = self.pretrained)

        self.local_image_gru = nn.GRU(input_size = self.input_size, hidden_size = self.output_size, 
                                        bidirectional = self.bidirectional, batch_first = True)

        self.concept_transform_layer = nn.Linear(self.output_size, self.output_size, bias = True)
        self.visual_transform_layer = nn.Linear(self.output_size, self.output_size, bias = True)
        self.concept_transform_layer_transpose = nn.Linear(self.output_size, self.output_size, bias = True)
        self.visual_transform_layer_transpose = nn.Linear(self.output_size, self.output_size, bias = True)

        # TODO: Add the layers corresponding to image captioning task
        # NOTE: For word attention
        self.embedding_layer = nn.Linear(self.vocab_size, self.word_hidden_size, bias = False)
        self.word_attention_layer = nn.Linear(self.word_hidden_size, self.output_size, bias = True)
        self.image_rep_attention_layer = nn.Linear(self.output_size, 1, bias = False)

        # NOTE: Gate for image rep
        self.gate_image_rep_layer = nn.Linear(self.output_size, 1, bias = True)
        self.image_rep_linear = nn.Linear(2 * self.output_size, self.output_size, bias = True)

        # TODO: Implement sentence generation and debug the code
        self.sentence_generation_gru = nn.GRU(self.output_size, self.output_size, bidirectional = False, batch_first = True)
        self.probability_dense = nn.Linear(self.output_size, self.vocab_size)


    def forward(self, images, word_embeddings):
        
        # vI
        concept_vector = self.concept_and_local_feature_predictor.concept_forward(images)
        # vc
        concept_set = ((concept_vector > torch.tensor(self.epsilon)).float() * 1).unsqueeze(-1) * torch.eye(self.output_size).reshape((1, 
                            self.output_size, self.output_size)).repeat(concept_vector.size(0), 1, 1).to(self.device)

        # NOTE: Local feature is of shape (7, 7, 512), but in the paper, it is (14, 14, 512)
        # Will check back later how to obtain conv5_4 feature map without max pooling
        
        # vl
        local_feature = self.concept_and_local_feature_predictor.local_feature_forward(images)
        local_feature = local_feature.view((local_feature.shape[0], local_feature.shape[1], -1))

        local_feature_out, local_feature_hidden = self.local_image_gru(local_feature)
        local_feature_out = local_feature_out.view(local_feature_out.size(0), local_feature.size(1), 2, self.output_size)
        local_feature_out = local_feature_out.reshape((2, local_feature_out.size(0), local_feature.size(1), -1))
        
        # vl_prime ==> bs x 512 x 512
        visual_representation = local_feature_out[0] + local_feature_out[1]

        # vI_prime ==> bs x 1024
        final_image_rep = self.semantic_guided_attention(concept_set, visual_representation)

        # TODO: Write code for image captioning, i.e., word attention and sentence generation. 
        # NOTE: In this function, return concept vector for concept prediction loss
        initial_hidden_state = torch.zeros(word_embeddings.size()[0], self.output_size).to(self.device)
        
        all_probability_distributions = []
        for idx, word_embedding in enumerate(word_embeddings): 
            probability_distribution = self.get_probability_distribution(word_embedding, final_image_rep[idx], visual_representation[idx], initial_hidden_state[idx])
            all_probability_distributions.append(probability_distribution)

        all_probability_distributions = torch.stack(all_probability_distributions).squeeze()

        return concept_vector, all_probability_distributions

    def get_probability_distribution(self, word_embedding, final_image_rep, visual_representation, initial_hidden_state):
        
        # hidden_states_and_output = {"hidden_states": [], "outputs": []}
        probability_distribution = []
        prev_hidden_state = initial_hidden_state

        for one_hot_vector in word_embedding:
            word_vector = self.embedding_layer(one_hot_vector)
            word_related_region_rep = self.word_guided_attention(visual_representation, prev_hidden_state)
            gated_image_rep = self.gated_image_representation(final_image_rep, prev_hidden_state)
            gru_hidden_state = word_related_region_rep + gated_image_rep + prev_hidden_state

            # NOTE: Some dimensionality problem or misunderstanding of paper. Read the paper carefully. :)))
            gru_output = self.sentence_generation_gru(word_vector.unsqueeze(0).unsqueeze(0), gru_hidden_state.unsqueeze(0).unsqueeze(0))

            # TODO: This part is looking erroneous. Look at it again. Problem in understanding the paper. :)
            prob_dist = self.probability_dense(gru_output[0].squeeze(0))
            probability_distribution.append(prob_dist)
            prev_hidden_state = gru_output[1].squeeze(0).squeeze(0)

        probability_distribution = torch.stack(probability_distribution)

        return probability_distribution

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

    def word_guided_attention(self, visual_representation, prev_hidden_state):
        
        attention_rep = self.image_rep_attention_layer(visual_representation).T + self.word_attention_layer(prev_hidden_state)
        
        attention_weights = torch.tanh(attention_rep)
        attention_weights = attention_weights / torch.sum(attention_weights, dim = 1)
        
        word_related_region_rep = torch.sum(visual_representation * attention_weights.T, dim = 1)
        return word_related_region_rep

    def gated_image_representation(self, image_rep, prev_hidden_state):
        gate = self.gate_image_rep_layer(prev_hidden_state)
        image_rep = self.image_rep_linear(image_rep)
        # print(image_rep.size())
        return gate * image_rep

class VisualQuestionAnsweringModel(nn.Module):

    def __init__(self, model_name, input_size = 49, hidden_size = 2048, output_size = 512, 
                pretrained = False, bidirectional = True, epsilon = 0.6, device = "cpu",
                vocab_size = 7000, word_hidden_size = 512, num_classes = 4):
        super(VisualQuestionAnsweringModel, self).__init__()

        self.model_name = model_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretrained = pretrained
        self.bidirectional = bidirectional
        self.epsilon = epsilon
        self.device = device

        self.vocab_size = vocab_size
        self.word_hidden_size = word_hidden_size
        self.num_classes = num_classes

        self.concept_and_local_feature_predictor = ConceptAndLocalFeaturePredictor(self.model_name,
                    hidden_size = self.hidden_size, output_size = self.output_size, pretrained = self.pretrained)

        self.local_image_gru = nn.GRU(input_size = self.input_size, hidden_size = self.output_size, 
                                        bidirectional = self.bidirectional, batch_first = True)

        self.concept_transform_layer = nn.Linear(self.output_size, self.output_size, bias = True)
        self.visual_transform_layer = nn.Linear(self.output_size, self.output_size, bias = True)
        self.concept_transform_layer_transpose = nn.Linear(self.output_size, self.output_size, bias = True)
        self.visual_transform_layer_transpose = nn.Linear(self.output_size, self.output_size, bias = True)

        self.embedding_layer = nn.Linear(self.vocab_size, self.word_hidden_size, bias = False)
        self.question_gru = nn.GRU(self.word_hidden_size, self.output_size, bidirectional = False, batch_first = True)
        
        self.question_attention_layer = nn.Linear(self.word_hidden_size, self.output_size, bias = False)
        self.image_rep_attention_layer = nn.Linear(self.output_size, self.output_size, bias = False)


        self.final_image_rep_attention_layer = nn.Linear(2 * self.output_size, self.output_size, bias = False)
        self.question_guided_rep_attention_layer = nn.Linear(self.output_size, self.output_size, bias = True)
        self.answer_distribution_layer = nn.Linear(self.output_size, self.num_classes)

    def forward(self, images, word_embeddings, question_embedding):
        
        # vI
        concept_vector = self.concept_and_local_feature_predictor.concept_forward(images)
        # vc
        concept_set = ((concept_vector > torch.tensor(self.epsilon)).float() * 1).unsqueeze(-1) * torch.eye(self.output_size).reshape((1, 
                            self.output_size, self.output_size)).repeat(concept_vector.size(0), 1, 1).to(self.device)

        # NOTE: Local feature is of shape (7, 7, 512), but in the paper, it is (14, 14, 512)
        # Will check back later how to obtain conv5_4 feature map without max pooling
        
        # vl
        local_feature = self.concept_and_local_feature_predictor.local_feature_forward(images)
        local_feature = local_feature.view((local_feature.shape[0], local_feature.shape[1], -1))

        local_feature_out, local_feature_hidden = self.local_image_gru(local_feature)
        local_feature_out = local_feature_out.view(local_feature_out.size(0), local_feature.size(1), 2, self.output_size)
        local_feature_out = local_feature_out.reshape((2, local_feature_out.size(0), local_feature.size(1), -1))
        
        # vl_prime ==> bs x 512 x 512
        visual_representation = local_feature_out[0] + local_feature_out[1]

        # vI_prime ==> bs x 1024
        final_image_rep = self.semantic_guided_attention(concept_set, visual_representation)
        question_rep = self.embedding_layer(question_embedding)
        question_rep_out, question_rep_hidden = self.question_gru(question_rep)

        question_related_region_rep = self.question_guided_attention(visual_representation, question_rep_hidden)
        answer_distribution = self.joint_embedding_and_classifier(final_image_rep, question_related_region_rep)

        return concept_vector, answer_distribution

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

    def question_guided_attention(self, visual_representation, question_rep_hidden):
        
        question_rep = self.question_attention_layer(question_rep_hidden)
        question_rep = question_rep.view(question_rep.size()[1], question_rep.size()[0], -1)
        image_rep = self.image_rep_attention_layer(visual_representation)

        res = torch.exp(torch.bmm(question_rep, image_rep)).squeeze()
        attention_weights = (res / torch.sum(res, dim = 0)).unsqueeze(2)
        
        question_related_region_rep = torch.sum(visual_representation * attention_weights, dim = 1)
        return question_related_region_rep

    def joint_embedding_and_classifier(self, final_image_rep, question_related_region_rep):

        attention_rep = torch.tanh(self.final_image_rep_attention_layer(final_image_rep) + \
                            self.question_guided_rep_attention_layer(question_related_region_rep))
        answer_distribution = nn.functional.softmax(self.answer_distribution_layer(attention_rep), dim = 1)

        return answer_distribution