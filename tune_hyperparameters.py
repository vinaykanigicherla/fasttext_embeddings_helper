import argparse
from embeddings_tuner.grid_search import grid_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Fasttext Unsupervised Learning")
    parser.add_argument("--search_space_path", help="Path to YAML file describing search space", 
                        default="data/def_hyp_search_space.yaml")
    parser.add_argument("--save_folder", help="Folder to which to save hyperparameter tuning results", 
                        default="hyp")
    
    required = parser.add_argument_group("Required arguments")
    required.add_argument('--data_path', help="Path to Training Data", required=True)
    required.add_argument('--analogies_path', help="Path to Analogies CSV File", required=True)
    args = parser.parse_args()

    grid_search(args.data_path, args.search_space_path, 
                args.analogies_path, args.save_folder)

