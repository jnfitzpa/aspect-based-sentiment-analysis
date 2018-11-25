# Aspect-Based Sentiment Analysis

The purpose of this repository is to create an aspect-based sentiment analysis (ABSA) model which involves a couple of steps. First, a review or comment is split up into topics using an unsupervised topic clustering algorithm. Then each sentence in the review is assigned to one of the topics and the sentiment for that sentence is evaluated. This allows us to assign an average sentiment for each topic.

wget https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz

gunzip amazon_reviews_us_Electronics_v1_00.tsv.gz
mkdir Data
mv amazon_reviews_us_Electronics_v1_00.tsv Data/amazon_reviews_us_Electronics_v1_00.tsv
