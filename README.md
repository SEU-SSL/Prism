# Prism

Prism is a multi-view learning approach for smart contract vulnerability detection. It learns complementary syntactic, control-flow, and data-flow information from aligned Abstract Syntax Tree (AST), Control Flow Graph (CFG), and Data Flow Graph (DFG) representations.

Prism normalizes and annotates contract code, uses a recursive neural network for the AST and graph attention networks for the CFG and DFG, and combines predictions from seven view combinations through full-stacking fusion.
