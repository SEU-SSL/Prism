# Prism

## Method

Prism performs binary vulnerability detection for smart contracts. It models two complementary views of each contract:

- The source-code view extracts an abstract syntax tree (AST), control-flow graph (CFG), and data-flow graph (DFG) from Solidity source code.

- The bytecode view extracts opcode sequences, a CFG, and a DFG from compiled EVM instructions.

- The two views are processed by graph attention networks (GATs). During training, the source-code model acts as the teacher and knowledge distillation transfers its information to the bytecode model. Predictions from seven view configurations are then fused to produce the final classification.

The main entry point is `NewMain.py`, and the model implementations are in `gnnmodels/`. The train/test split is controlled by `ParameterConfig.dataset_split_ratio`, which defaults to 0.2 for the test set.

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
