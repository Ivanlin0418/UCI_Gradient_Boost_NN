# Gradient Boosting Model with Stacking Ensemble

This project implements a machine learning pipeline using gradient boosting models and a stacking ensemble approach. It leverages XGBoost, LightGBM, and CatBoost classifiers to predict outcomes based on sparse input data. The project also includes feature engineering, dynamic feature selection, and hyperparameter optimization using Optuna. Please make sure you have a GPU with CUDA support to run this project (I'm using a RTX 3060ti).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contact](#contact)

## Installation

To run this project, you need to have Python installed along with the required packages. You can install the dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your data**: Ensure your data is in the correct sparse format as expected by the `load_sparse_data` function.

2. **Run the script**: Execute the main script to train the model and evaluate its performance.

```bash
python cuda-gradient-boost-model.py
```

## Dataset

The dataset used in this project is the [a5a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
## Contact
For questions or feedback, please contact Ivan Lin at il9082@rit.edu.
