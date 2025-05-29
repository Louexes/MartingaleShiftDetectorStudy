import optuna
import pandas as pd
from datetime import datetime
from copy import deepcopy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from optuna.pruners import MedianPruner
from optuna.trial import TrialState
import os
import argparse
from PBRS_LabelShift_Evaluation import evaluate_PBRS_FPR, evaluate_PBRS_TPR
from utils.utilities import (
    get_model,
    evaluate,
    load_cifar10_label_shift,
    run_martingale,
    load_clean_then_corrupt_sequence
)
from protector import get_protector_from_ents

# Define corruption types
CORRUPTIONS = (
    "shot_noise",
    "motion_blur",
    "snow",
    "pixelate",
    "gaussian_noise",
    "defocus_blur",
    "brightness",
    "fog",
    "zoom_blur",
    "frost",
    "glass_blur",
    "impulse_noise",
    "contrast",
    "jpeg_compression",
    "elastic_transform",
)

def preload_cifar10_data():
    """Preload CIFAR10 data to avoid multiprocessing issues"""
    print("Preloading CIFAR10 data...")
    transform = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2471, 0.2435, 0.2616)
    )
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transform
        ])
    )
    return dataset

class BasicDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

def run_optuna_pbrs_optimization(
    model,
    load_cifar10_corruption,
    load_cifar10_label_shift,
    BasicDataset,
    run_martingale,
    protector_factory,
    transform,
    args,
    device,
    corruption_types,
    severities,
    num_classes_list=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    seeds_tpr=range(3),
    seeds_fpr=range(3),
    n_trials=50,
    log_path='optuna_pbrs_results.csv',
    n_jobs=8
):
    """
    Optimize PBRS hyperparameters using Optuna.
    Fixed parameters:
    - gamma = 1/(8*sqrt(3))
    - eps_clip = 1.8
    """
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    # Fixed hyperparameters
    gamma = 1 / (8 * np.sqrt(3))
    eps_clip = 1.8

    def objective(trial):
        try:
            # Set device
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
            
            # Get model
            model = get_model(device)

            # Only tune buffer size and confidence threshold
            buffer_size = trial.suggest_categorical("buffer_size", [32, 64, 128, 256, 512, 1024])
            confidence_threshold = trial.suggest_float("confidence_threshold", 0.5, 0.95, step=0.05)

            run_id = f"{run_time}_B{buffer_size}_T{confidence_threshold:.2f}_G{gamma:.3f}_E{eps_clip:.2f}"
            print(f"\n🔧 Trial {trial.number}: {run_id}")

            # Create model copies for TPR and FPR evaluation
            model_copy_tpr = deepcopy(model)
            model_copy_fpr = deepcopy(model)

            # Create protector with fixed parameters
            args_protector = type('Args', (), {'gamma': gamma, 'eps_clip': eps_clip, 'device': device})()
            protector_tpr = protector_factory(model_copy_tpr, args_protector)
            protector_fpr = protector_factory(model_copy_fpr, args_protector)

            # Evaluate TPR
            tpr_result = evaluate_PBRS_TPR(
                model=model_copy_tpr,
                load_clean_then_corrupt_sequence=load_cifar10_corruption,
                BasicDataset=BasicDataset,
                run_martingale=run_martingale,
                protector=protector_tpr,
                transform=transform,
                args=args,
                device=device,
                corruption_types=corruption_types,
                severities=severities,
                seeds=seeds_tpr,
                buffer_capacity=buffer_size,
                confidence_threshold=confidence_threshold,
                num_classes=10,
                use_pbrs=True,
                log_path=f'tpr_results_{run_id}.csv'
            )

            # Calculate TPR metrics
            tpr_scores = [v['detection_rate'] for v in tpr_result.values()]
            delays = [v['avg_delay'] for v in tpr_result.values() if v['avg_delay'] is not None]
            mean_tpr = float(pd.Series(tpr_scores).mean())
            mean_delay = float(pd.Series(delays).mean()) if delays else 4000

            # Early pruning based on TPR
            if mean_tpr < 0.5:  # If TPR is too low, prune the trial
                raise optuna.TrialPruned()

            # Evaluate FPR
            fpr_result = evaluate_PBRS_FPR(
                model=model_copy_fpr,
                load_cifar10_label_shift=load_cifar10_label_shift,
                BasicDataset=BasicDataset,
                run_martingale=run_martingale,
                protector=protector_fpr,
                transform=transform,
                args=args,
                device=device,
                seeds=seeds_fpr,
                buffer_capacity=buffer_size,
                confidence_threshold=confidence_threshold,
                num_classes_list=num_classes_list,
                use_pbrs=True,
                log_path=f'fpr_results_{run_id}.csv'
            )

            # Calculate FPR metrics
            fpr_scores = list(fpr_result.values())
            mean_fpr = float(pd.Series(fpr_scores).mean())

            # Early pruning based on FPR
            if mean_fpr > 0.5:  # If FPR is too high, prune the trial
                raise optuna.TrialPruned()

            # Compute objective score with adjusted weights
            score = (0.9 * (1 - mean_fpr)) + (0.7 * mean_tpr) - (0.4 * (mean_delay / 4000))
            
            # Store metrics for this trial
            trial.set_user_attr("fpr", mean_fpr)
            trial.set_user_attr("tpr", mean_tpr)
            trial.set_user_attr("delay", mean_delay)

            results.append({
                'run_id': run_id,
                'trial': trial.number,
                'buffer_size': buffer_size,
                'confidence_threshold': confidence_threshold,
                'gamma': gamma,
                'eps_clip': eps_clip,
                'mean_fpr': mean_fpr,
                'mean_tpr': mean_tpr,
                'avg_delay': mean_delay,
                'score': score
            })

            print(f"🔎 Trial {trial.number}: score = {score:.3f} | FPR={mean_fpr:.3f} | TPR={mean_tpr:.3f} | Delay={mean_delay:.0f}")
            return score

        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
            raise optuna.TrialPruned()

    # Create study with pruning
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1
    )
    
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner
    )

    # Set up parallel processing for M1 Max
    if torch.backends.mps.is_available():
        mp.set_start_method('fork', force=True)  # Use 'fork' instead of 'spawn' for macOS
    
    # Run optimization with multiprocessing but without tqdm progress bar
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False  # Disable tqdm progress bar
    )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(log_path, index=False)
    
    # Print summary
    print(f"\nOptuna results saved to {log_path}")
    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.3f}")
    print(f"  Params: {study.best_trial.params}")
    print(f"  FPR: {study.best_trial.user_attrs['fpr']:.3f}")
    print(f"  TPR: {study.best_trial.user_attrs['tpr']:.3f}")
    print(f"  Delay: {study.best_trial.user_attrs['delay']:.0f}")
    
    # Print pruning statistics
    pruned_trials = len([t for t in study.trials if t.state == TrialState.PRUNED])
    print(f"\nPruning Statistics:")
    print(f"  Pruned trials: {pruned_trials}")
    print(f"  Completed trials: {len(study.trials) - pruned_trials}")
    
    return study

def main():
    # Set up arguments
    args = type("Args", (), {})()
    args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    args.batch_size = 64
    args.n_examples = 1000

    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Define normalization transform
    transform = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2471, 0.2435, 0.2616)
    )

    # Preload CIFAR10 data
    preload_cifar10_data()

    # Load model
    print("Loading model...")
    model = get_model(device)
    model.eval()

    # Load clean CIFAR-10 for source entropies
    print("Loading clean CIFAR-10 for source entropies")
    clean_ds = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,  # Set to False since we preloaded
        transform=transforms.Compose([
            transforms.ToTensor(),
            transform
        ])
    )
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False)
    source_ents, _, _, _ = evaluate(model, clean_loader, device)

    # Create protector factory function
    def protector_factory(model, args):
        return get_protector_from_ents(source_ents, args)

    # Run optimization
    study = run_optuna_pbrs_optimization(
        model=model,
        load_cifar10_corruption=load_clean_then_corrupt_sequence,
        load_cifar10_label_shift=load_cifar10_label_shift,
        BasicDataset=BasicDataset,
        run_martingale=run_martingale,
        protector_factory=protector_factory,
        transform=transform,
        args=args,
        device=device,
        corruption_types=CORRUPTIONS,
        severities=[1, 2, 3, 4, 5],
        n_trials=40,
        n_jobs=8
    )

if __name__ == "__main__":
    main()

