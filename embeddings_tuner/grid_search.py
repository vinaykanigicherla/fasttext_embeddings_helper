import pandas as pd 
import fasttext 
import yaml
import itertools
import ast 
import os 

from pathlib import Path 
from embeddings_tuner.evaluate import compute_recall, compute_mrr

def grid_search(data_path, hyp_search_space_path, analogies_path, save_folder):
    best_mrr, best_recall = 0, 0
    best_mrr_params, best_recall_params = {}, {}

    save_folder = Path(save_folder)
    logs_path = save_folder/"logs.txt" 
    
    if not save_folder.exists():
        os.mkdir(save_folder)

    print(f"Starting Hyperparameter Search. Saving results to {save_folder}. "
                + f"Logging results to {logs_path}.")

    with open(logs_path, "w") as f:
        f.write(f"Logs of hyperparameter search for FastText word" 
                + f"embedding model trained on {data_path} and tested on {analogies_path}")
    
    hyp_search_space = {}
    with open(hyp_search_space_path) as f:
        hyp_search_space = yaml.load(f, Loader=yaml.FullLoader) 

    keys, values = zip(*hyp_search_space.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for n, exp in enumerate(experiments):
        exp_path = save_folder/f"exp{n}"
        os.mkdir(exp_path)
        print(f"Experiment {n}:")
        print(f"Parameter Space: {exp}")

        print("Training model...")
        model = fasttext.train_unsupervised(data_path, **exp)
        model.save_model(str(exp_path/f"exp{n}.bin"))
        
        print("Calculating Metrics...")
        pred_df = make_preds(model, analogies_path)
        pred_df.to_csv(exp_path/f"exp{n}_preds.csv") 
        recall_10, mrr = compute_score(pred_df)
        print(f"Recall@10: {recall_10}, MRR: {mrr}")

        if mrr > best_mrr:
            best_mrr, best_mrr_params = mrr, exp
        if recall_10 > best_recall:
            best_recall, best_recall_params = recall_10, exp 

        with open(logs_path, "a") as f:
            f.write(f"\n\n\n Experiment {n}: \n")
            f.write(f"Parameter Space: {exp} \n")
            f.write(f"Recall@10: {recall_10}, MRR: {mrr}")
    
    print(f"Best Recall@10: {best_recall} | Best MRR: {best_mrr}")
    print(f"Hyperparameters for best Recall@10: {best_recall_params}")
    print(f"Hyperparameters for best MRR: {best_mrr_params}")

    with open(save_folder/"best_hyperparameters_mrr.yaml", 'w') as f:
        yaml.dump(best_mrr_params, f)


    with open(save_folder/"best_hyperparameters_recall.yaml", 'w') as f:
        yaml.dump(best_recall_params, f)

    

def make_preds(model, analogies_path):
    data = pd.read_csv(analogies_path)
    c1, c2, c3, c4 = data.columns
    data["target"] = data[c4].apply(lambda x: str([x]))
    data["pred"] = None

    for idx, row in data.iterrows():
        ft_analogies = model.get_analogies(row[c1], row[c2], row[c3])
        row["pred"] = str([a[1] for a in ft_analogies]) 

    return data     
    
def compute_score(data):
    sum_recall_10 = 0
    sum_mrr = 0
    
    for row in data.iterrows():
        row = row[1]
        act = ast.literal_eval(row["target"])
        pred = ast.literal_eval(row["pred"])
        sum_recall_10 += compute_recall(act, pred, 10)
        sum_mrr += compute_mrr(act, pred)

    recall_10 = sum_recall_10 / len(data)
    mrr = sum_mrr / len(data)
    print('Recall@10:', recall_10)
    print('MRR:', mrr)
    return recall_10, mrr
