# Aspect-Based Sentiment Analysis

The purpose of this repository is to create an aspect-based sentiment analysis (ABSA) model which involves a couple of steps. First, a review or comment is split up into topics using an unsupervised topic clustering algorithm. Then each sentence in the review is assigned to one of the topics and the sentiment for that sentence is evaluated. This allows us to assign an average sentiment for each topic.

## Setting Up The Environment

Please install Anaconda if it isn't already. Next, navigate to the local cloned repository and type the following in the linux shell:

* conda env create -f environment.yml

After the environment has been loaded, activate the environment with the following comand:

* source activate tensorflow

## Training Sentiment Models

To train the sentiment models, first you need to download the Amazon review dataset from s3. After cloning the repo and browsing to the local directory, type the following commands in the linux shell:

* wget https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz

* gunzip amazon_reviews_us_Electronics_v1_00.tsv.gz

* mkdir Data

* mv amazon_reviews_us_Electronics_v1_00.tsv Data/amazon_reviews_us_Electronics_v1_00.tsv

Now the sentiment models can be trained using the following commands:

* python train_sent_bow.py

* python train_sent_cnn.py

