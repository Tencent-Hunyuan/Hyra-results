# Qubit Routing: SWAP minimization

Insert SWAP gates so every two-qubit gate of a circuit acts on physically
adjacent qubits, minimizing the added CNOT overhead (1 SWAP = 3 CNOTs).

**Benchmark:** RevLib reversible-logic, QFT, and Ising circuits routed onto three
device topologies: a 20-qubit lattice (**Q20**), a 105-qubit Willow-style device
(**Willow**), and a 156-qubit IBM-Heron-style device (**Heron**).

`solution.json` holds the **61 validated routed circuits** (each `output_circuit`
in OpenQASM 3.0).

## Result: CNOTs added (lower is better)

Compared with SimpleTES (Table 2, gpt-oss-120b) of *Evaluation-driven Scaling for
Scientific Discovery* ([arXiv:2604.19341](https://arxiv.org/abs/2604.19341)):

| Topology | SimpleTES | Hyra |
|---|--:|--:|
| Q20 (20 q) | 45,441 | **37,869** |
| Willow (105 q) | 96,774 | **95,898** |
| Heron (156 q) | 126,822 | **124,602** |
| **Total** | **269,037** | **258,369** |

Hyra adds fewer CNOTs than SimpleTES on every topology and overall: **258,369
vs 269,037** (−10,668).
