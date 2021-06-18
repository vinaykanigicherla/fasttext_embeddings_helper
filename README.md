# FastText Embeddings Helper
This repository contains Python scripts to perform Hyperparameter Tuning and Visualization for FastText's Unsupervised word embedding model. 

More specifically, it conducts grid search hyperparameter tuning using a user provided search space (in the form of a YAML file) and a user provided 
four-column analogies dataset. It chooses the hyperparameters that yield the best Recall@10 and MRR metrics calculated from predictions on the provided 
analogy dataset using the FastText models `get_analogies()` function. 

Additionally, we can create either a 2D or 3D PCA visualization of the word vectors in the model's vocabulary. Given a 2-column word pair dataset 
(such as a word similarity dataset), we can also create low dimensional visualizations of these pairs based on the vectors learned by the model.   

## Setup 
1. Clone this repository 
```
$ git clone https://github.com/vinaykanigicherla/fasttext_embeddings_helper.git
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
Hyperparameters are tuned using a grid search, where the best combination of values is selected based on the MRR and Recall@10 metrics the model achieves on 
the provided analogies dataset. The training file provided must be a data file which is an appropriate input to a FastText model. The analogies file 
provided must be a CSV file have four columns. Assuming these columns respectively are A, B, C, and D, each row must be an example of the analogy A:B = C:D. Trained models and their respective prediction files are saved to their own folders within the hyperparameter tuning save folder. All runs are logged to `logs.txt`, hyperparameters that yield the best MRR are saved to `best_hyperparameters_mrr.yaml`, and hyperparameters that yield the best Recall@10 are saved to `best_hyperparameters_recall.yaml`, 

To tune hyperparameters, providing path to training data file and analogies data file is required:
```
$ python tune_hyperparameters.py --data_path data/train.txt --analogies_path data/analogies.csv
```
The user can also provide a custom hyperparameter search space in the form of a YAML file. The default search space file `def_hyp_search_space.yaml` is as follows:
```
dim:
- 100
- 300
loss:
- ns
- hs
lr:
- 0.05
- 0.1
minCount:
- 1
model:
- cbow
- skipgram
```
To use custom hyperparameter search space, set the `search_space_path` flag to the path of your custom YAML file. To change the name of the save folder from the default (`hyp/`) set the `save_folder` flag to the desired name:
```
$ python tune_hyperparameters.py --data_path data/train.txt --analogies_path data/analogies.csv --search_space_path data/custom_hyp_ss.yaml --save_folder tuning_results
```

## Visualization 
The visualization script can be used for the PCA visualization of all the word vectors in the FastText model's vocabulary. It can also be used for the visualization of a specific set of word pairs. To visualize a specific set of word pairs, a pairs dataset must be provided in the form of a 2-column CSV file where each corresponds to one pair. Ensure that all the words within this data file are present in the model's vocabulary. 

To visualize all the word vectors in the model's vocabulary, set the `model_path` flag to the path to the saved model binary file:
```
$ python visualize.py --model_path models/model.bin 
```
To visualize pairs of words, also set the `pairs_path` flag to the path of the word pairs data file:
```
$ python visualize.py --model_path models/model.bin --pairs_path data/pairs.csv 
```
For visualizations in 3D instead of 2D, set the `is_3d` flag to `True`:
```
$ python visualize.py --model_path models/model.bin --is_3d True 
$ python visualize.py --model_path models/model.bin --pairs_path data/pairs.csv --is_3d True 
```

