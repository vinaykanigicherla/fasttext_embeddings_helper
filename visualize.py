import argparse
import pandas as pd
from fasttext import load_model
from embeddings_tuner.visualization_funcs import plot_all_embeddings, plot_pair_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Fasttext Unsupervised Learning")
    parser.add_argument("--pairs_path", help="Path to item pairs CSV file", type=str, 
                        default=None)
    parser.add_argument("--is_3d", help="Option to create 3D Visualizations", type=bool, 
                        default=False)

    required = parser.add_argument_group("Required Arguments")
    required.add_argument('--model_path', help="Path to Fasttext model binary", type=str, required=True)
    args = parser.parse_args()

    model = load_model(args.model_path)

    if not args.pairs_path:
        plot_all_embeddings(model, args.is_3d)
    else:
        pairs = pd.read_csv(args.pairs_path)
        plot_pair_points(pairs, model, args.is_3d)


