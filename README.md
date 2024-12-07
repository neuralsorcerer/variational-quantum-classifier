# Variational Quantum Classifier for Binary Classification

This repository provides an implementation of a simple Variational Quantum Classifier (VQC) using [PennyLane](https://pennylane.ai/) and Python. The project demonstrates how to:

1. Generate and preprocess a synthetic binary classification dataset.
2. Encode classical data into quantum states.
3. Define and train a parameterized quantum circuit (ansatz) as a classifier.
4. Apply gradient-based optimization to minimize a loss function and improve model accuracy.
5. Visualize training metrics (loss and accuracy) over epochs.

## Project Overview

Quantum Machine Learning (QML) blends quantum computing concepts with machine learning techniques. A Variational Quantum Classifier (VQC) is a hybrid approach where classical optimization routines tune the parameters of a quantum circuit to make predictions. By encoding classical data into quantum states and adjusting the circuit parameters, a VQC may leverage quantum effects such as superposition and entanglement, potentially achieving enhanced performance on certain tasks.

This project implements a basic VQC for a two-dimensional synthetic classification task. It includes:

- **Data Generation & Normalization:** Using scikit-learn’s `make_classification` to create a 2D dataset and normalizing features.
- **Quantum Circuit Construction:** A simple ansatz is built using parameterized single-qubit rotations and entangling CNOT gates.
- **Training Loop with Gradient Descent:** PennyLane’s `GradientDescentOptimizer` updates circuit parameters to reduce the mean squared error loss.
- **Performance Metrics & Visualization:** Tracking loss, training accuracy, and test accuracy over epochs, and plotting results using `matplotlib`.

## Key Files

- `main.ipynb`: A Jupyter notebook containing the code and explanations.
- `requirements.txt`: A list of required packages.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- [PennyLane](https://pennylane.ai/) (for quantum simulations and gradient computations)
- [NumPy](https://numpy.org/) (for numerical operations)
- [scikit-learn](https://scikit-learn.org/) (for generating synthetic datasets)
- [matplotlib](https://matplotlib.org/) (for data visualization)

Install dependencies:

```bash
pip install pennylane numpy scikit-learn matplotlib
```

### Running the Project

#### Clone this repository:

```bash
git clone https://github.com/neuralsorcerer/variational-quantum-classifier.git
cd variational-quantum-classifier
```

#### Open the Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

#### Run all cells in the notebook to:

1. Generate the dataset.
2. Initialize and train the variational quantum circuit.
3. Display the training loss and accuracy plots.

### Interpreting the Results

#### Loss Plot:

Shows how the mean squared error on the training set changes over training epochs. A decreasing trend suggests that the model is learning.

#### Accuracy Plots:

Compare training and test set accuracies over epochs. Improving accuracy indicates that the quantum classifier is correctly learning to distinguish between the two classes. Test accuracy provides insight into the model’s generalization capabilities.

### Customization and Next Steps

#### Ansatz Customization:

Try modifying the parameterized circuit structure by adding more layers, using different sets of gates, or varying the connectivity pattern.

#### Optimization Techniques:

Experiment with different optimizers (e.g., Adam, RMSProp) or vary the learning rate.

#### Data Encoding:

Explore alternative encoding methods (e.g., amplitude encoding or different embedding strategies) to represent the classical data in the quantum state more effectively.

#### Barren Plateaus & Noise Considerations:

Implement techniques to avoid barren plateaus (e.g., layerwise training) or simulate realistic noise models to test on near-term quantum devices.

## License

This project is licensed under the [MIT License](LICENSE).
