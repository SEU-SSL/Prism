# Prism

## Method

Prism is a multi-view learning approach for smart contract vulnerability detection. It preserves complementary information from the Abstract Syntax Tree (AST), Control Flow Graph (CFG), and Data Flow Graph (DFG), deriving all three views from a shared normalized AST so that their nodes remain aligned.

The method has three stages:

1. **Preprocessing.** Solidity source code is parsed into an AST. Identifiers are normalized, security-sensitive elements are annotated, and the annotations are propagated through the tree. The annotated AST is used to construct the AST, CFG, and DFG views.

2. **View-specific learning.** A recursive neural network encodes the hierarchical AST view, while Graph Attention Networks encode the CFG and DFG views. Prism trains seven base learners for the individual views and all non-empty view combinations.

3. **Full-stacking fusion.** The seven base learners generate out-of-fold predictions. A decision-tree meta-learner combines these predictions into the final vulnerability decision.

## Running the Project

1. Create a Python 3.8 environment and install the dependencies. The project is configured for CUDA 11.8 builds of PyTorch and DGL:

   ```bash

   python3.8 -m venv .venv

   source .venv/bin/activate

   python -m pip install --upgrade "pip<25.1"

   python -m pip install -r requirements.txt

   ```

2. Prepare the data directories. Place Solidity source files in `dataset/timestamp/` and preprocessed files in `dataset/TP/`:

   ```text

   dataset/

   ├── timestamp/              # Solidity source files

   └── TP/

       ├── dataset_S.json     # Source-code views and labels

       ├── dataset_B.json     # Bytecode views and labels

       ├── embs.npy            # Source-code token/node embeddings

       └── embb.npy            # Bytecode opcode embeddings

   ```

   `dataset_S.json`, `dataset_B.json`, and both embedding files must use the same sample order and consistent labels. If the data is stored elsewhere, update `FF` and `viewfile` in `ParameterConfig.py`.

3. Make Solidity compiler version `0.4.25` available. Importing `ParameterConfig.py` attempts to install and select this version.

4. Run training and evaluation from the repository root:

   ```bash

   python NewMain.py

   ```

   The original implementation uses `cuda:0`, so a compatible NVIDIA GPU and CUDA environment are required.

## Dataset Source

This repository does not include the original Solidity contracts, labels, preprocessed JSON files, or NPY embeddings. These files must be prepared separately. The code defines the expected interface and default locations: source files are read from `dataset/timestamp/`, while paired source-code and bytecode features are read from `dataset/TP/dataset_S.json`, `dataset/TP/dataset_B.json`, `dataset/TP/embs.npy`, and `dataset/TP/embb.npy`.

The repository files and configuration do not identify a public dataset name, download URL, or preprocessing script for these samples. To reproduce a paper or experiment dataset, obtain the original contracts and labels from the corresponding release and generate the files in the format described above; the default paths in this repository are not dataset download URLs.
