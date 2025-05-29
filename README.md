# Label Shift & Covariate Shift Detection Experiments

This repository provides a framework for evaluating methods to detect distribution shifts (label shift and covariate/covariate shift) in image classification, with a focus on the CIFAR-10 dataset. The codebase supports several state-of-the-art and baseline methods, robust experiment logging, and result visualization.

---

## Table of Contents

- [Label Shift \& Covariate Shift Detection Experiments](#label-shift--covariate-shift-detection-experiments)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Methods](#methods)
  - [Repository Structure](#repository-structure)
  - [Setup \& Installation](#setup--installation)
  - [Running Experiments](#running-experiments)
  - [Experiment Arguments](#experiment-arguments)
  - [Results \& Logging](#results--logging)
  - [Extending the Framework](#extending-the-framework)
  - [Troubleshooting](#troubleshooting)
  - [Citations](#citations)
  - [Contact](#contact)
  - [Weighted CDF Implementations](#weighted-cdf-implementations)
  - [PBRS Buffer Class and Additional Protector Functions](#pbrs-buffer-class-and-additional-protector-functions)

---

## Overview

The goal of this repository is to compare different statistical and learning-based methods for detecting when the test data distribution has shifted from the training distribution. The main focus is on:
- **Label Shift:** The class distribution changes between training and test data.
- **Covariate Shift:** The input distribution changes (e.g., via image corruptions).

Detection is performed using entropy-based martingale tests and various weighting/protection schemes.

---

## Methods

The following methods are implemented and can be selected via the `--method` argument:

- **baseline:** No adaptation; runs the martingale test on the raw entropy stream.
- **pbrs:** PBRS filtering (buffer and confidence threshold).
- **w-cdf:** Weighted CDF using importance weighting based on estimated label distributions.
- **w-bbse:** Black Box Shift Estimation (BBSE) for more accurate weighting.
- **w-bbseods:** BBSE with Online Distribution Shift (ODS) corrections.

Each method is evaluated for both False Positive Rate (FPR, under label shift) and True Positive Rate (TPR, under covariate shift).

---

## Repository Structure

```
.
├── Run_LabelShift_Experiments.py      # Main experiment runner
├── PBRS_LabelShift_Evaluation.py      # PBRS evaluation logic
├── WeightedCDF_LabelShift_Evaluation.py # Weighted CDF, BBSE, ODS logic
├── utilities.py                       # Data loading, model helpers, martingale, etc.
├── experiment_logger.py               # Logging and result saving
├── plotting.py                        # Plotting utilities
├── protector.py                       # Protector/CDF classes
├── utils_entropy_cache.py             # Entropy caching helpers
├── data/                              # CIFAR-10 and CIFAR-10-C data
├── reproduction.py                    # Script to reproduce main martingale-based shift detection results
├── DSD_Analysis.ipynb                 # Self-contained Jupyter notebook for in-depth analysis and visualization of shift detection
└── logs/                              # Experiment logs and results
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Install dependencies:**
   - Python 3.8+
   - PyTorch (with CUDA or MPS support if available)
   - torchvision
   - numpy, pandas, tqdm, matplotlib

   You can install the main dependencies with:
   ```bash
   pip install torch torchvision numpy pandas tqdm matplotlib
   ```

3. **Download CIFAR-10 and CIFAR-10-C:**
   - The code will automatically download CIFAR-10 if not present.
   - For CIFAR-10-C, download from [CIFAR-10-C GitHub](https://github.com/hendrycks/robustness) and place the files in `./data/CIFAR-10-C/`.

---

## Running Experiments

The main entry point is `Run_LabelShift_Experiments.py`. You can run experiments with different methods and settings using command-line arguments.

**Example: Run the baseline method**
```bash
python Run_LabelShift_Experiments.py --method baseline
```

**Example: Run PBRS with 5 seeds**
```bash
python Run_LabelShift_Experiments.py --method pbrs --seeds 0 1 2 3 4
```

**Example: Run weighted CDF**
```bash
python Run_LabelShift_Experiments.py --method w-cdf
```

**All arguments can be listed with:**
```bash
python Run_LabelShift_Experiments.py --help
```

---

## Experiment Arguments

Key arguments (with defaults):

- `--method`  
  Which method to use (`baseline`, `pbrs`, `w-cdf`, `w-bbse`, `w-bbseods`).

- `--batch_size`  
  Batch size for data loading (default: 64).

- `--device`  
  Device to use (`cuda`, `mps`, or `cpu`).

- `--buffer_capacity`  
  Buffer size for PBRS (default: 512).

- `--confidence_threshold`  
  Confidence threshold for PBRS (default: 0.5).

- `--seeds`  
  List of random seeds (default: `[0]`).

- `--number_classes`  
  List of class counts to test for label shift (default: `[1,2,3,4,5,6,7,8,9,10]`).

- `--corruptions`  
  List of corruption types for covariate shift (see code for defaults).

- `--severities`  
  List of severity levels for corruptions (default: `[1,2,3,4,5]`).

---

## Results & Logging

- **Logs and results** are saved in a timestamped directory under `./logs/`.
- **FPR and TPR results** are saved as CSV and JSON files.
- **Plots** (e.g., TPR comparison) are saved as PNG files.
- **Console output** provides progress, debug info, and summary statistics.

---

## Extending the Framework

- **Add new methods:**  
  Implement your method as a function and add it to the `METHODS` dictionary in `Run_LabelShift_Experiments.py`.

- **Add new datasets:**  
  Add new data loaders to `utilities.py` and update experiment logic as needed.

- **Change evaluation logic:**  
  Modify or extend the evaluation functions in the relevant files.

---

## Troubleshooting

- **Not enough samples for a class subset:**  
  The code oversamples with replacement to ensure enough data for each experiment.

- **Device errors (e.g., MPS):**  
  Some PyTorch operations may not be supported on MPS. The code includes workarounds for most cases.

- **Missing dependencies:**  
  Install all required Python packages as listed above.

- **Slow runs:**  
  Use a GPU (`cuda` or `mps`) for best performance.

---

## Citations

If you use this codebase in your research, please cite the relevant papers for PBRS, BBSE, and ODS methods as well as the CIFAR-10 and CIFAR-10-C datasets.

---

## Contact

For questions or contributions, please open an issue or pull request on the repository.

---

**Happy experimenting!**

---

## reproduction.py

This script reproduces the main results for martingale-based distribution shift detection on CIFAR-10 and CIFAR-10-C. It evaluates several adaptation methods (e.g., no adaptation, POEM, TENT) under various corruption types and severities.

**Usage:**
```bash
python reproduction.py --device cuda --methods in_dist no_adapt poem --corruption gaussian_noise --n_examples 1000 --batch_size 64
```

**Arguments:**
- `--device`: Device to use (`cuda`, `cpu`, etc.)
- `--methods`: List of methods to evaluate (`in_dist`, `no_adapt`, `poem`)
- `--corruption`: Corruption type (default: `gaussian_noise`)
- `--all_corruptions`: If set, evaluates all corruption types
- `--n_examples`: Number of examples per evaluation (default: 1000)
- `--batch_size`: Batch size for evaluation (default: 64)

**Output:**
- Saves a pickle file `martingale_behaviour.pkl` with martingale progression data.
- Prints accuracy summary for each method and corruption.
- Useful for reproducing the main results and figures in the paper.

---

## DSD_Analysis.ipynb

This Jupyter notebook provides a self-contained, in-depth analysis and visualization of distribution shift detection. It does **not** require outputs from other scripts (such as `martingale_behaviour.pkl`). Instead, it includes all necessary code to load, process, and visualize data relevant to shift detection experiments.

**Typical workflow:**
- Load and process data within the notebook (e.g., delay data, corruption types, method results).
- Analyze detection statistics, accuracy, and performance metrics.
- Visualize results across corruption types, severities, and adaptation methods.
- Generate publication-quality figures and tables.

**To use:**
1. Open `DSD_Analysis.ipynb` in Jupyter.
2. Follow the cells to load, analyze, and visualize the results. No prior experiment output is required.

---

## Weighted CDF Implementations

This repository provides several implementations of weighted cumulative distribution functions (CDFs) for robust distribution shift detection. These are used to reweight entropy statistics based on estimated label distributions, improving detection under label shift and covariate shift.

### utils/weighted_cdf.py
Implements a basic weighted CDF using importance weights derived from estimated label distributions on the source and target data. The weights are clipped for stability. Includes helper functions to estimate label and pseudo-label distributions from model predictions.

- **Class:** `WeightedCDF`
- **Purpose:** Computes a weighted CDF of entropy values, reweighting by the ratio of target to source class probabilities.
- **Usage:** Used for the `w-cdf` method in experiments.

### utils/weighted_cdf_bbse.py
Implements a weighted CDF using Black Box Shift Estimation (BBSE) to estimate the true target label distribution by inverting the confusion matrix. This approach provides more accurate weights under label shift.

- **Class:** `BBSEWeightedCDF`
- **Purpose:** Computes a weighted CDF with weights estimated via BBSE, correcting for label shift using the confusion matrix.
- **Helper Functions:**
  - `estimate_confusion_matrix`: Estimates the confusion matrix and source label marginals.
  - `estimate_target_distribution_from_preds`: Estimates the target label distribution from model predictions.
  - `estimate_shift_weights`: Full pipeline to compute BBSE weights.
- **Usage:** Used for the `w-bbse` method in experiments.

### utils/weighted_cdf_bbse_ods.py
Extends the BBSE approach with Online Distribution Shift (ODS) corrections, allowing the target distribution estimate to be updated online as new pseudo-labels are observed.

- **Class:** `BBSEODSWeightedCDF`
- **Purpose:** Computes a weighted CDF with BBSE and ODS, updating the target distribution and weights online.
- **Key Methods:**
  - `correct_p_test`: Corrects the target distribution using the confusion matrix.
  - `ods_update` / `batch_ods_update`: Online update of the target distribution with new pseudo-labels.
  - `update_weights`: Updates the sample weights after distribution changes.
- **Usage:** Used for the `w-bbseods` method in experiments.

These implementations are central to the label shift detection methods evaluated in this repository. For more details, see the code in the `utils/` directory and the main experiment scripts.

---

## PBRS Buffer Class and Additional Protector Functions

The `protector.py` file has been extended with a new `PBRSBuffer` class and additional helper functions to support various weighted CDF implementations. These additions enable robust distribution shift detection by managing a buffer of entropy samples and providing specialized protector instances.

### PBRSBuffer Class

The `PBRSBuffer` class manages a fixed-capacity buffer of entropy samples, ensuring balanced class representation. It is used in the PBRS (Pseudo-Balanced Reservoir Sampling) method to filter and store entropy values based on pseudo-labels.

- **Purpose:** Maintains a balanced buffer of entropy samples across classes, ensuring that each class is represented equally up to a target quota.
- **Key Methods:**
  - `accept(y_hat)`: Determines if a new sample should be accepted based on class representation.
  - `add(step_idx, entropy, y_hat)`: Adds a new sample to the buffer, evicting the oldest sample if the buffer is full.
  - `full()`: Checks if the buffer has reached its capacity.
  - `get_entropies()`: Returns the list of entropy values stored in the buffer.
  - `get_indexed_entropies()`: Returns a list of tuples containing the step index and entropy value.
  - `reset()`: Clears the buffer and resets class counts.

### Additional Protector Functions

The following functions have been added to `protector.py` to create specialized protector instances using different weighted CDF implementations:

#### get_weighted_protector_from_ents(source_ents, source_pseudo_labels, p_s, p_t, args)

- **Purpose:** Creates a `Protector` instance using a basic weighted CDF. The weights are derived from the estimated label distributions on the source and target data.
- **Parameters:**
  - `source_ents`: Entropy values from the source (clean) data.
  - `source_pseudo_labels`: Pseudo-labels for the source data.
  - `p_s`: Source label distribution (as a NumPy array or tensor).
  - `p_t`: Estimated target label distribution.
  - `args`: Namespace containing parameters like `gamma`, `eps_clip`, and `device`.
- **Usage:** Used for the `w-cdf` method in experiments.

#### get_bbse_weighted_protector_from_ents(source_ents, pseudo_labels, weights, args)

- **Purpose:** Creates a `Protector` instance using a weighted CDF with Black Box Shift Estimation (BBSE). The weights are estimated using the confusion matrix to correct for label shift.
- **Parameters:**
  - `source_ents`: Entropy values from the source data.
  - `pseudo_labels`: Pseudo-labels for the source data.
  - `weights`: Pre-computed BBSE weights.
  - `args`: Namespace containing parameters like `gamma`, `eps_clip`, and `device`.
- **Usage:** Used for the `w-bbse` method in experiments.

#### get_bbse_ods_weighted_protector_from_ents(source_ents, p_test, p_source, source_labels, confusion_matrix, ods_alpha, args)

- **Purpose:** Creates a `Protector` instance using a weighted CDF with BBSE and Online Distribution Shift (ODS) corrections. This allows the target distribution estimate to be updated online as new pseudo-labels are observed.
- **Parameters:**
  - `source_ents`: Entropy values from the source data.
  - `p_test`: Estimated target label distribution.
  - `p_source`: Source label distribution.
  - `source_labels`: True labels for the source data.
  - `confusion_matrix`: Estimated confusion matrix.
  - `ods_alpha`: Learning rate for online updates.
  - `args`: Namespace containing parameters like `gamma`, `eps_clip`, and `device`.
- **Usage:** Used for the `w-bbseods` method in experiments.

These additions enhance the flexibility and robustness of the distribution shift detection framework, allowing for more sophisticated methods to be evaluated and compared.