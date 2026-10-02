# Prism

## Method

Prism is a multi-view learning approach for smart contract vulnerability detection. It preserves complementary information from the Abstract Syntax Tree (AST), Control Flow Graph (CFG), and Data Flow Graph (DFG), deriving all three views from a shared normalized AST so that their nodes remain aligned.

The method has three stages:

1. **Preprocessing.** Solidity source code is parsed into an AST. Identifiers are normalized, security-sensitive elements are annotated, and the annotations are propagated through the tree. The annotated AST is used to construct the AST, CFG, and DFG views.

2. **View-specific learning.** A recursive neural network encodes the hierarchical AST view, while Graph Attention Networks encode the CFG and DFG views. Prism trains seven base learners for the individual views and all non-empty view combinations.

3. **Full-stacking fusion.** The seven base learners generate out-of-fold predictions. A decision-tree meta-learner combines these predictions into the final vulnerability decision.

#### Dataset

The evaluation dataset is the manually validated smart-contract dataset released by Luo et al. and used by SCVHunter. It was derived from SmartBugs through keyword-based candidate selection followed by manual labeling. The benchmark contains 1,200 vulnerability-specific Solidity contracts: 300 for each of the four vulnerability types.

## Quick start

```bash
python3.10 -m venv .venv

.venv/bin/python -m pip install torch==2.5.1

.venv/bin/python -m pip install -r requirements.txt
```

Prepare the Solidity sources, train the vulnerability-specific models, and predict one contract:

```bash
.venv/bin/python -m prism prepare --data dataset/scvhunter --output artifacts/prepared

.venv/bin/python -m prism train --prepared artifacts/prepared/prepared.json \

  --output artifacts/runs --device auto

.venv/bin/python -m prism predict --source path/to/contract.sol \

  --bundle artifacts/runs/reentrancy/seed-42/bundle.pt
```

Install the Solidity compiler versions declared by the source files before preprocessing. Use `--device cpu` for CPU execution or `--device cuda` for an available NVIDIA GPU.
