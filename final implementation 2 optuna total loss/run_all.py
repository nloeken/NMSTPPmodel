# run_pipeline.py
import gc
import os
import pandas as pd
import optuna
import torch
import argparse  # For command-line arguments

# --- Import project modules ---
from config import *
from data_loading import load_and_merge
from data_preprocessing import preprocess
from feature_engineering import add_features, build_sequences
from model import NMSTPP_Sequential
from model_training import train_model
from model_evaluation import evaluate_model
from utils import make_dataloaders

def run_training_pipeline():
    """
    (Pipeline Stage 3)
    Orchestrates the model training and hyperparameter optimization part of the pipeline.
    This function is called when the --train flag is used.
    """
    print("--- Stage 3: Starting Feature Engineering and Model Training ---")

    def objective(trial: optuna.Trial):
        """The main objective function for Optuna hyperparameter optimization."""
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        d_model = trial.suggest_categorical('d_model', [128, 256])
        num_layers = trial.suggest_int('num_layers', 2, 4)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])

        print(f"\n===== Starting Trial {trial.number} with params: =====")
        print(trial.params)

        print("Loading data from combined file for feature engineering...")
        df = pd.read_csv(COMBINED_FILE)
        if FAST_DEBUG and len(df) > FAST_MAX_ROWS:
            df = df.sample(n=FAST_MAX_ROWS, random_state=42).copy()

        df_featured = add_features(df)
        X, y, le_action_dict = build_sequences(df_featured)

        # --- FAST MODE: nur während Optuna Trials ---
        SAMPLE_FRAC = 0.1  # z.B. 10% benutzen
        if True:  # optional: trial.number < X  oder z.B. immer während optuna
            n_samples = int(len(y['zone']) * SAMPLE_FRAC)
            print(f"⚡ Fast Mode aktiv: reduziere {len(y['zone'])} → {n_samples} Sequenzen")

            # Wichtig: gleicher Index für alle Arrays verwenden
            X['cat'] = X['cat'][:n_samples]
            X['cont'] = X['cont'][:n_samples]
            X['360'] = X['360'][:n_samples]
            y['zone'] = y['zone'][:n_samples]
            y['act'] = y['act'][:n_samples]
            y['time'] = y['time'][:n_samples]

        del df, df_featured
        gc.collect()

        train_loader, val_loader, action_weights = make_dataloaders(X, y, batch_size)

        nhead = 8 if d_model % 8 == 0 else 4  # Ensure nhead is a divisor of d_model
        model = NMSTPP_Sequential(
            cont_dim=X['cont'].shape[2],
            dim_360=X['360'].shape[2],
            seq_len=SEQ_LEN,
            num_zones=len(ZONE_CENTROIDS),
            num_actions=len(le_action_dict),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        model = train_model(
            model, train_loader, val_loader, action_weights,
            lr=lr, weight_decay=weight_decay
        )

        macro_f1_score = evaluate_model(model, val_loader, le_action_dict)

        del model, X, y, train_loader, val_loader, action_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return macro_f1_score

    # --- Start Optuna Study ---
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Set number of trials here

    print("\n\n===== Optimization Complete =====")
    print(f"Best Macro F1-Score: {study.best_value:.4f}")
    print("Best Hyperparameters:", study.best_params)

def main():
    """
    Main entry point to run the entire pipeline.
    Use command-line flags to specify which stages to run.
    Example: python run_pipeline.py --load --preprocess --train
    """
    parser = argparse.ArgumentParser(description="Run parts of the NMSTPP data pipeline.")
    parser.add_argument('--load', action='store_true', help="Run Stage 1: Load and merge raw JSON data into individual CSVs.")
    parser.add_argument('--preprocess', action='store_true', help="Run Stage 2: Combine and preprocess CSVs into a single file.")
    parser.add_argument('--train', action='store_true', help="Run Stage 3: Feature engineering and model training.")
    args = parser.parse_args()

    if args.load:
        load_and_merge()

    if args.preprocess:
        preprocess()

    if args.train:
        run_training_pipeline()

    if not any(vars(args).values()):
        print("No stages selected. Use --load, --preprocess, or --train to run parts of the pipeline.")
        print("Example: python run_pipeline.py --load --preprocess --train")

if __name__ == '__main__':
    # This can help prevent some multiprocessing issues with PyTorch on Windows
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()