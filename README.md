# FastText Embeddings Helper
This repository contains Python scripts to perform Hyperparameter Tuning and Visualization for FastText's Unsupervised word embedding model. 

More specifically, it conducts grid search hyperparameter tuning using a user provided search space (in the form of a YAML file) and a user provided 
four-column analogies dataset. It chooses the hyperparameters that yield the best Recall@10 and MRR metrics calculated from predictions on the provided 
analogy dataset using the FastText models `get_analogies()` function. 

Additionally, we can create either a 2D or 3D PCA visualization of the word vectors learned by the model. Given a 2-column word pair dataset 
(such as a word similarity dataset), we can also create low dimensional visualizations of these pairs based on the vectors learned by the model.   
