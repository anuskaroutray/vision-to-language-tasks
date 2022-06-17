# Vision-to-Language tasks based on Attributes and Attention Mechanism

Many academics have become interested in vision-to-language projects, which try to combine computer vision and natural language processing. They encode images into feature representations and decode them into natural language words in usual techniques. High-level semantic notions and nuanced interactions between image regions and natural language parts are ignored. This research attempts to make full use of these data by utilising textguided attention (TA) and semantic-guided attention (SA) to locate more linked spatial information and close the semantic gap between vision and language. Two-level attention networks are used in the paper's strategy. The text-guided attention network, for example, is used to choose text-related regions. The SA network, on the other hand, is used to emphasise concept-related locations and concept-related concepts. Finally, all of this data is combined to give captions or responses. Image captioning and visual question answering studies have been carried out in practise, with the experimental findings demonstrating the proposed approach's superior performance.

[Paper](https://ieeexplore.ieee.org/document/8718014)

### File Description

This repository contains 3 python scripts and 3 directories. 

- [main.py](./run.py): Main script to aggregate all the functions, train the Image Captioning Model and Visual Question Answering Model, as well as evaluate them.

The following files are present in the directory ```data```.
- [dataloader.py](./data/dataloader.py): Script to create a custom Dataloader for the image and text datasets (MS COCO, Flickr30k, Flickr8k and Toronto COCO-QA). The class ```ImageCaptionDataset``` and ```VisualQuestionAnsweringDataset``` preprocesses the text and applies required transforms on the images.
- [preprocess_data.py](./data/preprocess_data.py): Script to create a uniform json for each of MS COCO, Flickr30k, Flickr8k and Toronto COCO-QA data. The raw data for all four datasets are in different directory structure. So, to maintain uniformity, run ```preprocess_data.py``` to generate the required ```.json``` files.

The following file is present in the directory ```src```.
- [model.py](./src/model.py): Script to create different modules for the task of Image Captioning and Visual Question Answering.
- [utils.py](./src/utils.py): Script with helper functions for vocab.py
- [vocab.py](./src/vocab.py): Script to create required vocabulary compatible with PyTorch 

The following files are present in the directory ```utils```.
- [env.py](./utils/env.py): Script to define the global random seed environment for the sake of reproducibility.
- [loss.py](./utils/loss.py): Script to define the loss function as Image Captioning Loss and Visual Question Answering Loss.
- [trainer.py](./utils/test.py): Script contains the function to train and evaluate the models.

### Setup the environment  

```
  conda env create -f retrieval.yml
  conda activate retrieval
```

### Data preprocessing

```./fetch_datasets.sh```
This will obtain the Flickr30k, Flickr8k, MS-COCO and Toronto COCO-QA dataset in the required format for training and evaluation. 
NOTE: 
- Images of Flickr30k dataset need to be requested through a form available on the official [website](http://shannon.cs.illinois.edu/DenotationGraph/), hence the above script would not be able to fetch the images of Flickr30k dataset.
- Since MS-COCO dataset has sizes in the range of GB (13 GB for train split, 6GB for validation split and 12GB for test split), running this script would require a couple of hours. 