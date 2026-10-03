# Prism

## Method

Prism is a multi-view learning approach for smart contract vulnerability detection. It preserves complementary information from the Abstract Syntax Tree (AST), Control Flow Graph (CFG), and Data Flow Graph (DFG), deriving all three views from a shared normalized AST so that their nodes remain aligned.

The method has three stages:

1. **Preprocessing.** Solidity source code is parsed into an AST. Identifiers are normalized, security-sensitive elements are annotated, and the annotations are propagated through the tree. The annotated AST is used to construct the AST, CFG, and DFG views.

2. **View-specific learning.** A recursive neural network encodes the hierarchical AST view, while Graph Attention Networks encode the CFG and DFG views. Prism trains seven base learners for the individual views and all non-empty view combinations.

3. **Full-stacking fusion.** The seven base learners generate out-of-fold predictions. A decision-tree meta-learner combines these predictions into the final vulnerability decision.

## Dataset

The evaluation dataset is the manually validated smart-contract dataset released by Luo et al. and used by SCVHunter. It was derived from SmartBugs through keyword-based candidate selection followed by manual labeling. The benchmark contains 1,200 vulnerability-specific Solidity contracts: 300 for each of the four vulnerability types.

## Quick Start

From the `Prism` directory, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Then prepare the data and start training:

```bash
python NewMain.py prepare --data dataset/scvhunter --task reentrancy --output artifacts/prepared

python NewMain.py train --prepared artifacts/prepared.json --task reentrancy --device auto --epochs 1 --output artifacts/run
```
