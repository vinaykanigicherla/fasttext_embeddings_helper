# FastText Embeddings Helper
This repository contains Python scripts to perform Hyperparameter Tuning and Visualization for FastText's Unsupervised word embedding model. 

More specifically, it conducts grid search hyperparameter tuning using a user provided search space (in the form of a YAML file) and a user provided 
four-column analogies dataset. It chooses the hyperparameters that yield the best Recall@10 and MRR metrics calculated from predictions on the provided 
analogy dataset using the FastText models `get_analogies()` function. 

Additionally, we can create either a 2D or 3D PCA visualization of the word vectors learned by the model. Given a 2-column word pair dataset 
(such as a word similarity dataset), we can also create low dimensional visualizations of these pairs based on the vectors learned by the model.   

## Setup 
1. Clone this repository and enter the directory
```
$ git clone 
$ cd fasttext_embeddings_helper
```
2. Install FastText Python Module
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ sudo pip install .
```
3. Install remaining requirements
```
$ pip install -r requirements.txt
```

## Usage
### Hyperparameter Tuning
Run `tune_hyperparameters.py` providing path to training data file and analogies data file:
```
$ python tune_hyperparameters.py --data_path data/train.txt --analogies_path data/analogies.csv
```
To use custom hyperparameter search space, provide path to YAML file to `search_space_path` flag. To set custom save folder name, provide appropriate name
to `save_folder` flag:
```
$ python tune_hyperparameters.py --data_path data/train.txt --analogies_path data/analogies.csv --search_space_path data/custom_hyp_ss.yaml --save_folder tuning_results
```
