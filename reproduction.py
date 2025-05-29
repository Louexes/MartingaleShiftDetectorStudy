import os
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

from utilities import *
from plotting import *
import protector as protect
import tent, poem
#from tent import Tent, configure_model, collect_params
from temperature_scaling import ModelWithTemperature
from utils.cli_utils import softmax_ent

CORRUPTIONS = (
    "shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise",
    "defocus_blur", "brightness", "fog", "zoom_blur", "frost", "glass_blur",
    "impulse_noise", "contrast", "jpeg_compression", "elastic_transform"
)

def og_get_model(method: str, device: str, protector=None):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    model = ModelWithTemperature(model, 1.8).to(device)

    if method == 'no_adapt':
        return model.eval()
    elif method == 'poem':
        model = poem.configure_model(model)
            
    if method == "tent":
        model = tent.configure_model(model)
        #params, _ = tent.collect_params(model)
        #optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9)
        #model = tent.Tent(model, optimizer)
    return model.eval()


def og_run_martingale(entropy_streams: Dict[str, np.ndarray], protector) -> Dict[str, Dict[str, list]]:
    results = {}
    for name, z_seq in entropy_streams.items():
        protector.reset()
        logs, eps = [], []
        for z in z_seq:
            u = protector.cdf(z)
            protector.protect_u(u)
            logs.append(np.log(protector.martingales[-1] + 1e-8))
            eps.append(protector.epsilons[-1])
        results[name] = {"log_sj": logs, "eps": eps}
    return results


def og_evaluate(model, dataloader, device):
    """Evaluate entropy values, accuracy, logits and labels."""
    entropies, correct, total = [], 0, 0
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            entropies.extend(softmax_ent(logits).tolist())
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            logits_list.append(logits.cpu())
            labels_list.append(y.cpu())
    accuracy = correct / total
    return np.array(entropies), accuracy, logits_list, labels_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--methods', type=list, default=['in_dist', 'no_adapt', 'poem'], choices=['no_adapt', 'in_dist', 'poem'])
    parser.add_argument('--corruption', type=str, default='gaussian_noise')
    parser.add_argument('--all_corruptions', action='store_true')
    parser.add_argument('--n_examples', type=int, default=1000) # use 3500 to reproduce Bar et al.
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))

    print("Loading clean CIFAR-10 as source entropy")
    clean_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(), transform]))
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False)
    corruptions = CORRUPTIONS if args.all_corruptions else [args.corruption]

    records = []
    acc_summary = {}

    for method in args.methods:
        print(f"\nMethod: {method}")
        # Get results on source distribution
        model = og_get_model(method, device)
        source_ents, source_acc, _, _ = og_evaluate(model, clean_loader, device)

        protector = protect.get_protector_from_ents(
            source_ents,
            argparse.Namespace(gamma=1 / (8 * np.sqrt(3)), eps_clip=1.8, device=device)
        )

        # Use POEM wrapper if method is 'poem'
        if method == 'poem':
            params, _ = poem.collect_params(model)
            optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9)
            model = poem.POEM(model, optimizer, protector)

        # Can add TENT wrapper here

        # Track accuracy and entropy
        accs, entropy_streams = {}, {}

        if method == 'in_dist':
        # Clean data
            entropy_streams['clean_s0'] = source_ents[:args.n_examples]
            accs['clean_s0'] = source_acc
        else:
        # Corrupted evaluations
            for corruption in corruptions:
                for severity in range(1,6):
                    print(f"Evaluating {corruption}, severity {severity}")
                    x, y = load_cifar10c(args.n_examples, severity, corruption)
                    dataset = BasicDataset(x, y, transform=transform)
                    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

                    ents, acc, _, _ = og_evaluate(model, loader, device)
                    key = f"{corruption}_s{severity}"
                    entropy_streams[key] = ents
                    accs[key] = acc

        results = og_run_martingale(entropy_streams, protector)

        for key, martingale_dict in results.items():
            corruption, severity_str = key.rsplit('_s', 1)
            severity = int(severity_str)
            records.append({
                'exp_type': 'martingale',
                'model': 'resnet50_gn_timm',
                'corruption': corruption,
                'level': severity,
                'method': method,
                'seed': 2024,
                'martingales': martingale_dict['log_sj'],
                'epsilons': martingale_dict['eps'],
            })

        for key, acc in accs.items():
            acc_summary[f"{method}/{key}"] = acc

    # Save data
    prog_df = pd.DataFrame(records)
    prog_df.to_pickle("martingale_behaviour.pkl")
    print("\nSaved martingale progression data to disk.")

    # Print accuracy summary
    print("\nAccuracy summary:")
    for key, acc in acc_summary.items():
        print(f"{key}: {acc * 100:.2f}%")


if __name__ == '__main__':
    main()
