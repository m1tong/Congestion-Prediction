# DSC-Capstone-B07-Congestion-Prediction

## Description
Welcome to the DSC180A Capstone Project: Graph ML for Chip Profiling. This report presents a novel approach to early congestion prediction in Integrated Circuit (IC) design through the use of Graph Attention Networks (GATs). Addressing the challenge of congestion, we developed an interpretable graph neural network model. Utilizing the NCSU-DigIC-GraphData dataset, which comprises 13 unique netlists with distinct congestion characteristics, our methodology includes feature engineering, graph embedding, and a two layer GAT convolutional architecture. Initial results, based on one netlist, show promising training and test loss metrics, though the model tends to predict average congestion values, indicating room for improvement

<a href='https://m1tong.github.io/DSC180-Website/project/'><img src='https://img.shields.io/badge/Read_Full_Project-HTML-green'> <a href='https://drive.google.com/file/d/1l02sLVGAcwbpovdyrLqEiBVG49FuiT-B/view'><img src='https://img.shields.io/badge/Report-PDF-blue'> <a href='https://drive.google.com/file/d/1cUXHOLFV43pKeQYA4pugsL4v1YYz9t08/view'><img src='https://img.shields.io/badge/Poster-PDF-red'>


## Table of Contents

1. [Installation](#installation)
2. [Conda Environment](#conda-environment)
3. [Instructions on How to Run the Code](#instructions)
4. [Links](#links)

### Installation

Provide instructions on how to install and set up the project. Include any prerequisites or dependencies.

```bash
# Clone the repository
git clone https://github.com/m1tong/Congestion-Prediction.git

# Navigate to the project directory
cd Congestion-Prediction

# Create and activate Conda environment
conda env create -f environment.yml
conda activate DSC180-B07
```

### Conda Environment
Environment Name: DSC180-B07 <br>
Environment File: environment.yml

### Instructions
1. cd to the folder that this model is in
2. Run the line `python models.py`
```bash
python models.py
```
4. There will be a command line asking whether you want to run gat or gcn model
5. Type in either `gcn` or `gat` then the model will start running

### Links
Website: [Here](https://m1tong.github.io/DSC180-Website/)





