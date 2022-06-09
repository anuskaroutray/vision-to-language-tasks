# Import the required libraries
import requests
import urllib.request
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import os, json, argparse, csv

from typing import Union, Iterable
import torchtext
from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
from src.vocab import build_vocab_from_iterator
import torch
import torch.nn.functional as F

import regex as re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess(data_path, dataset_name = "coco", split = "all"):
    preprocessed_dict = {}

    if dataset_name == "coco":
        data = json.load(open(data_path, "r"))

        if split == 'test':
            raise ValueError("'test' split has no annotations")

        for image_dict in tqdm(data["images"], position = 0, leave = True):
            Id = image_dict["id"]
            image_path = image_dict["file_name"]
            
            if "train" in image_path:
                dir_path = "./datasets/coco/train2014"
            elif "val" in image_path:
                dir_path = "./datasets/coco/val2014"
            preprocessed_dict[Id] = {"image_path": os.path.join(dir_path, image_path), "captions": []}

        for annotation_dict in data["annotations"]:
            image_id = annotation_dict["image_id"]
            caption = annotation_dict["caption"]
            if image_id in preprocessed_dict:
                preprocessed_dict[image_id]["captions"].append(caption)

    elif dataset_name == "flickr30k":
        data = json.load(open(data_path, "r"))
        dir_path = "./datasets/flickr30k/flickr30k-images"
        
        for image_dict in tqdm(data["images"], position = 0, leave = True):
            
            Id = image_dict["imgid"]
            image_path = image_dict["filename"]
            
            if split in [image_dict["split"], "all"]:
            
                preprocessed_dict[Id] = {"image_path": os.path.join(dir_path, image_path), "captions": []}
                sentences_list = image_dict["sentences"]
            
                for token_sent_dict in sentences_list:
                    image_id = token_sent_dict["imgid"]
                    caption = token_sent_dict["raw"]
                    preprocessed_dict[image_id]["captions"].append(caption)

    elif dataset_name == "flickr8k":

        with open('./datasets/flickr8k/Flickr_8k.trainImages.txt') as f:
            train_images = f.readlines()
            for i in range(len(train_images)):
                train_images[i] = train_images[i][:-1]
        with open('./datasets/flickr8k/Flickr_8k.testImages.txt') as f:
            test_images = f.readlines()
            for i in range(len(test_images)):
                test_images[i] = test_images[i][:-1]
        with open('./datasets/flickr8k/Flickr_8k.devImages.txt') as f:
            dev_images = f.readlines()
            for i in range(len(dev_images)):
                dev_images[i] = dev_images[i][:-1]

        with open(data_path) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            
            token_list = line.split()
            Id = i//5 + 1
            image_file = token_list[0][:-2]

            if split == "train" and image_file not in train_images:
                continue
            if split == "test" and image_file not in test_images:
                continue
            if split == "val" and image_file not in dev_images:
                continue

            image_path = os.path.join("./datasets/flickr8k/Flicker8k_Dataset", image_file)
            caption = (" ").join(token_list[1:])
            if i % 5 == 0:
                preprocessed_dict[Id] = {"image_path": image_path, "captions": []}
            preprocessed_dict[Id]["captions"].append(caption)

    elif dataset_name == "toronto-cocoqa":

        data = json.load(open(data_path, "r"))
        train_questions = open("datasets/toronto-cocoqa/train/questions.txt", "r").readlines()
        train_answers = open("datasets/toronto-cocoqa/train/types.txt", "r").readlines()
        train_image_ids = open("datasets/toronto-cocoqa/train/img_ids.txt", "r").readlines()

        val_questions = open("datasets/toronto-cocoqa/test/questions.txt", "r").readlines()
        val_answers = open("datasets/toronto-cocoqa/test/types.txt", "r").readlines()
        val_image_ids = open("datasets/toronto-cocoqa/test/img_ids.txt", "r").readlines()

        ques_ans_dict = {}

        if "train" or "all" in split:
            for i in range(len(val_image_ids)):
                ques_ans_dict[int(train_image_ids[i][:-1])] = {"question": preprocess_caption(train_questions[i][:-1]), 
                                                                "answer": int(train_answers[i][:-1])}
        elif "val" in split:
            for i in range(len(val_image_ids)):
                ques_ans_dict[int(val_image_ids[i][:-1])] = {"question": preprocess_caption(val_questions[i][:-1]),
                                                             "answer": int(val_answers[i][:-1])}       

        if split == 'test':
            raise ValueError("'test' split has no annotations")

        for image_dict in tqdm(data["images"], position = 0, leave = True):
            Id = image_dict["id"]
            image_path = image_dict["file_name"]
            
            if "train" in image_path:
                dir_path = "./datasets/coco/train2014"
            elif "val" in image_path:
                dir_path = "./datasets/coco/val2014"
            
            if Id not in ques_ans_dict.keys():
                continue

            preprocessed_dict[Id] = {"image_path": os.path.join(dir_path, image_path), 
                                    "question": ques_ans_dict[Id]["question"],
                                    "answer": ques_ans_dict[Id]["answer"],
                                    "captions": []}

        for annotation_dict in data["annotations"]:
            image_id = annotation_dict["image_id"]
            caption = annotation_dict["caption"]
            if image_id in preprocessed_dict:
                preprocessed_dict[image_id]["captions"].append(caption)

    return preprocessed_dict

def get_word_frequency(words):

    word_freq = dict()

    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

    return word_freq

# NOTE: Used to build the general vocabulary
def preprocess_caption(caption):

    caption = re.sub('[^a-zA-Z0-9.?,]', ' ', caption)
        
    url = re.compile(r'https?://\S+|www\.\S+')
    caption = url.sub(r'', caption)
    
    html = re.compile(r'<.*?>')
    caption = html.sub(r'', caption)

    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
    caption = emoji_pattern.sub(r'', caption)

    return caption

# Used for concept vocabulary
def preprocess_caption_for_concept(caption, tokenizer):

    caption = re.sub('[^a-zA-Z0-9.?,]', ' ', caption)
        
    url = re.compile(r'https?://\S+|www\.\S+')
    caption = url.sub(r'', caption)
    
    html = re.compile(r'<.*?>')
    caption = html.sub(r'', caption)

    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
    caption = emoji_pattern.sub(r'', caption)

    stop_words = set(stopwords.words('english')) 

    tokens = tokenizer(caption) 
    filtered_caption = [] 
    
    for w in tokens: 
        if w not in stop_words: 
            filtered_caption.append(w) 

    filtered_caption = " ".join(filtered_caption) 

    return filtered_caption

def build_vocabulary(dataset = "coco", split = "train"):

    preprocessed_dict = json.load(open(f'./datasets/{dataset}/{split}_image_captions.json', 'r'))

    corpus = []
    for image_id, path_caption_dict in preprocessed_dict.items():
        captions = preprocessed_dict[image_id]["captions"]
        for i in range(len(captions)):
            captions[i] = preprocess_caption(captions[i])
        corpus.extend(captions)
    
    # Tokenizer: Converts a string or a sentence into a list of words.
    tokenizer = get_tokenizer("basic_english")
    tokens = [tokenizer(caption) for caption in corpus]

    vocabulary = build_vocab_from_iterator(tokens, min_freq = 5, specials = ("<UNK>"), specials_first = True)

    torch.save(vocabulary, f'./datasets/{dataset}/{split}_vocabulary.pth')

def build_concept_vocabulary(dataset = "coco", split = "all", num_concepts = 512):

    preprocessed_dict = json.load(open(f'./datasets/{dataset}/{split}_image_captions.json', 'r'))
    tokenizer = get_tokenizer("basic_english")

    corpus = []
    for image_id, path_caption_dict in preprocessed_dict.items():
        captions = preprocessed_dict[image_id]["captions"]
        for i in range(len(captions)):
            captions[i] = preprocess_caption_for_concept(captions[i], tokenizer)
        corpus.extend(captions)
    
    tokens = [tokenizer(caption) for caption in corpus]

    vocabulary = build_vocab_from_iterator(tokens, max_tokens = num_concepts)
    torch.save(vocabulary, f'./datasets/{dataset}/{split}_concept_vocabulary.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'toronto-cocoqa', type = str, choices=['coco', 'flickr8k', 'flickr30k', 'toronto-cocoqa'])
    parser.add_argument('--data_path', type = str)
    parser.add_argument('--split', default = 'train', type = str, choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--preprocess', default = False, type = bool)
    parser.add_argument('--get_vocab', default = False, type = bool)
    parser.add_argument('--build_concept_vocab', default = False, type = bool)
    parser.add_argument('--num_concepts', default = 512, type = int)

    args = parser.parse_args()

    if args.preprocess:
        preprocessed_dict = preprocess(args.data_path, args.dataset, split = args.split)
        save_path = open(f'./datasets/{args.dataset}/{args.split}_image_captions.json', 'w')
        json.dump(preprocessed_dict, save_path)
    
    if args.get_vocab:
        build_vocabulary(dataset = args.dataset, split = args.split)
        # NOTE: To use the vocabulary to get one hot vector, use following line of code
        #           one_hot = get_one_hot_word_vector(vocabulary, ["studio", "constructions"])

    if args.build_concept_vocab:
        build_concept_vocabulary(dataset = args.dataset, split = args.split)