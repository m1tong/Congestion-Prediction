# DSC-Capstone-B07-Congestion-Prediction

## Description
Welcome to the DSC180A Capstone Project: Graph ML for Chip Profiling. This report presents a novel approach to early congestion prediction in Integrated Circuit (IC) design through the use of Graph Attention Networks (GATs). Addressing the challenge of congestion, we developed an interpretable graph neural network model. Utilizing the NCSU-DigIC-GraphData dataset, which comprises 13 unique netlists with distinct congestion characteristics, our methodology includes feature engineering, graph embedding, and a two layer GAT convolutional architecture. Initial results, based on one netlist, show promising training and test loss metrics, though the model tends to predict average congestion values, indicating room for improvement

## Table of Contents

1. [Installation](#installation)
2. [Conda Environment](#conda-environment)

### Installation

Provide instructions on how to install and set up the project. Include any prerequisites or dependencies.

```bash
# Clone the repository
git clone https://github.com/m1tong/Congestion-Prediction.git

# Navigate to the project directory
cd Congestion-Prediction

# Create and activate Conda environment
conda env create -f environment.yml
conda activate capstone
```

### Conda Environment
Environment Name: DSC180-B07 <br>
Environment File: environment.yml



