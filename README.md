# Optimizer Stability Benchmark

This repository contains code and experiments for the research project:

**Numerical Optimization in Machine Learning: Stability and Robustness of Optimizers Under Noise and Precision Constraints**


##  Overview
We benchmark five popular optimizers — **SGD, SGD with Momentum, Adam, RMSProp, and Adagrad** — under challenging training conditions:
- **Label Noise**: 0%, 10%, and 30% injected into MNIST labels
- **Precision**: FP32 vs FP64
- **Datasets**: MNIST classification + toy sine regression
- **Models**: Small CNN (MNIST) and MLP (sine regression)

We evaluate:
- Convergence speed  
- Final test accuracy (mean ± std across seeds)  
- Variance across runs  
- Stability (divergence/failure rates)  
- Precision gap (difference FP32 vs FP64)  

##  Repository Structure

optimizer-stability/

│

├── src/ # Source code

│ ├── data.py # Dataset loaders and label noise injection

│ ├── model.py # MLP and CNN definitions

│ ├── train.py # Training loop for experiments

│ ├── analyze.py # Aggregates results, creates summary.csv

│ ├── plot_learning_curves.py # Generates learning curve plots

│ └── init.py

│

├── results/ # Results will be stored here

│ ├── light/ # Quick experiments (few epochs)

│ └── regression/ # Toy regression dataset

│

├── run_grid_light.ps1 # PowerShell script to run experiments

├── precision_gap.py # Script to compute precision differences

├── charts_convergence.py # Example plotting script

│

├── requirements.txt # Python dependencies

├── README.md # Project description (this file)

└── Numerical Method Research paper.docx # Draft of the research paper


Installation & Setup

1. Clone the repo:
   
   git clone https://github.com/yourusername/optimizer-stability.git
   cd optimizer-stability
   
Create virtual environment:


python -m venv .venv

.\.venv\Scripts\activate   # On Windows

Install dependencies:

pip install -r requirements.txt

Running Experiments

Run a grid of experiments on MNIST:


.\run_grid_light.ps1
Run a single experiment:


python -m src.train --optimizer adam --lr 0.001 --noise 0.1 --precision float32 --seed 0 --epochs 3 --outdir results\light --dataset mnist
 Analysis & Plots
Aggregate results:


python -m src.analyze --indir results\light
Compute precision gaps:

python precision_gap.py
Generate learning curves:

python -m src.plot_learning_curves --indir results\light
Example Results
Final Test Accuracy (MNIST, 0% noise, Adam FP32): ~98.9%

Precision Gap (FP64 vs FP32): < 0.2%

Stability: No divergence observed across runs

Research Paper
Draft manuscript is included as:


Numerical Method Research paper.docx
