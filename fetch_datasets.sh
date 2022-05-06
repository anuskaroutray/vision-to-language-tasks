#!/bin/bash

mkdir datasets
cd datasets
mkdir flickr30k
mkdir coco
mkdir flickr8k

# fetch flickr8k
cd flickr8k
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
unzip Flickr8k_Dataset > flickr8k-dataset
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
unzip Flickr8k_text > flickr8k-text
cd ..

# fetch flickr30k
cd flickr30k
wget https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip
unzip flickr30k.zip > flickr30k-captions
cp -r /home/bt2/18CS10069/retrieval/datasets/flickr30k/flickr30k-images ./
cd ..

# fetch coco
cd coco
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2014.zip
wget http://images.cocodataset.org/zips/test2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2015.zip
wget http://images.cocodataset.org/zips/test2015.zip

unzip train2014.zip
unzip annotations_trainval2014.zip
unzip val2014.zip
unzip image_info_test2014.zip
unzip test2014.zip
unzip image_info_test2015.zip
unzip test2015.zip
cd ..

# run data preprocessing flickr8k
python data/preprocess_data.py --dataset flickr8k --data_path datasets/flickr8k/Flickr8k.token.txt --split train --preprocess True --get_vocab True 
python data/preprocess_data.py --dataset flickr8k --data_path datasets/flickr8k/Flickr8k.token.txt --split test --preprocess True  
python data/preprocess_data.py --dataset flickr8k --data_path datasets/flickr8k/Flickr8k.token.txt --split val --preprocess True  
python data/preprocess_data.py --dataset flickr8k --data_path datasets/flickr8k/Flickr8k.token.txt --split all --preprocess True  

# run data preprocessing flickr30k
python data/preprocess_data.py --dataset flickr30k --data_path datasets/flickr30k/flickr30k/dataset.json --split train --preprocess True --get_vocab True 
python data/preprocess_data.py --dataset flickr30k --data_path datasets/flickr30k/flickr30k/dataset.json --split test --preprocess True   
python data/preprocess_data.py --dataset flickr30k --data_path datasets/flickr30k/flickr30k/dataset.json --split val --preprocess True   
python data/preprocess_data.py --dataset flickr30k --data_path datasets/flickr30k/flickr30k/dataset.json --split all --preprocess True  

# run data preprocessing coco
python data/preprocess_data.py --dataset coco --data_path datasets/coco/annotations/captions_val2014.json --split val --preprocess True  
python data/preprocess_data.py --dataset coco --data_path datasets/coco/annotations/captions_val2014.json --split train --preprocess True --get_vocab True
python data/preprocess_data.py --dataset coco --data_path datasets/coco/annotations/captions_val2014.json --split all --preprocess True  