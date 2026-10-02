# Prism: Each Intermediate Representation Matters

Prism is a multi-view learning approach for smart contract vulnerability detection. It learns complementary features from aligned Abstract Syntax Tree (AST), Control Flow Graph (CFG), and Data Flow Graph (DFG) views using a recursive neural network and graph attention networks. Full-stacking fusion combines predictions from seven view configurations through a decision-tree meta-classifier.

## Usage.

Create and activate a Python 3.8 environment from the repository root:

```bash
python3.8 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade "pip<25.1"

python -m pip install -r requirements.txt

python -m pip check
```

The requirements target Linux x86_64 with CPython 3.8 and CUDA 11.8 builds of PyTorch and DGL. The original training code contains explicit CUDA calls and requires a working NVIDIA GPU. Python 3.8 is used for the available `glove-python-binary` wheel.

- Configure the project root, preprocessed-data directory and training settings in `ParameterConfig.py`.

- Place Solidity sources under `dataset/timestamp/`. This is the source path used by the original `NewMain.py` entry point.

- Provide aligned `dataset_S.json`, `dataset_B.json`, `embs.npy` and `embb.npy` under `dataset/TP/`, or change `ParameterConfig.viewfile` accordingly. The source and bytecode datasets must use the same sample order and labels.

- The original configuration installs and selects Solidity compiler `0.4.25` when imported. Ensure that version is available and compatible with your sources.

- Run the original entry point from the repository root:

```bash
python NewMain.py
```

Datasets, embeddings and pretrained checkpoints are not bundled. This repository preserves the original research sources; dependency installation does not establish that the full training workflow or paper metrics have been reproduced.

`Mythril .py` is an independent baseline example that invokes the external `myth` command. Its dependencies are isolated from the learning environment:

```bash
python3.8 -m venv .venv-mythril

.venv-mythril/bin/python -m pip install -r requirements-mythril.txt
```

Activate that separate environment when running the Mythril example.

## requirements

- torch==2.0.1+cu118

- dgl==1.1.3+cu118

- torch-geometric==2.3.1

- numpy==1.24.4

- scipy==1.10.1

- scikit-learn==1.3.2

- gensim==4.3.2

- smart-open==6.4.0

- glove-python-binary==0.2.0

- networkx==3.1

- anytree==2.8.0

- py-solc-x==1.1.1

- solcast==1.3.0

- evm-cfg-builder==0.3.1

- pyevmasm==0.2.3

`glove-python-binary` supplies `glove`, `py-solc-x` supplies `solcx`, and `scikit-learn` supplies `sklearn`. Standard-library modules and local files under `gnnmodels/` are not separate pip packages. The optional Mythril dependency is listed in `requirements-mythril.txt`.
