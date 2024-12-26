# Water Demand Forecasting Preprocessing Guide

This guide will walk you through the steps to set up your environment, configure the necessary files, and run the preprocessing script for the water demand forecasting project. This guide assumes you are new to coding, so each step is explained in detail.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- Python (version 3.6 or higher)
- Git (for cloning the repository)

## Step 1: Clone the Repository

First, clone the repository to your local machine using Git. Open your terminal and run the following command:

```sh
git clone https://github.com/oscarwoodi/water-demand-forecasting-neural-networks.git
cd water-demand-forecasting-neural-networks
```

## Step 2: Set Up a Virtual Environment

Setting up a virtual environment helps manage dependencies and avoid conflicts. Run the following commands in your terminal:

```sh
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

## Step 3: Install Required Packages

With the virtual environment activated, install the required packages using `pip`. Run the following command:

```sh
pip install -r requirements.txt
```

## Step 4: Configure the `config.yml` File

The `config.yml` file contains paths and settings needed for preprocessing. Open the `config.yml` file in a text editor and ensure it contains the following configuration:

```yaml
# Relevant file/directory paths
PATHS:
  RAW_DATA_DIR: 'data/raw/'
  RAW_DATASET: 'data/raw/data_raw.csv'
  RAW_EXOG_DIR: 'data/raw_exog/'
  RAW_EXOG_DATASET: 'data/raw/exog_raw.csv'

  PREPROCESSED_DATA: 'data/preprocessed/preprocessed.csv'
  PREPROCESSED_DWT_DATA: 'data/preprocessed/dwt/preprocessed.csv'
  PREPROCESSED_CEEMDAN_DATA: 'data/preprocessed/ceemdan/preprocessed.csv'
  PREPROCESSED_MODEL_RESIDUAL_DATA: 'data/preprocessed/residuals/sarima/sarima_16wk.csv'

  MODELS: 'results/models/'
  DATA_VISUALIZATIONS: 'img/data_visualizations/'
  FORECAST_VISUALIZATIONS: 'img/forecast_visualizations/'
  EXPERIMENT_VISUALIZATIONS: 'img/experiment_visualizations/'
  INTERPRETABILITY_VISUALIZATIONS: 'img/interpretability_visualizations/'
  LOGS: 'results/logs/'

  PREDICTIONS: './results/predictions/'
```

Ensure the paths match the structure of your project directory.

## Step 5: Run the Preprocessing Script

With the configuration file set up, you can now run the preprocessing script. In your terminal, navigate to the top of the directory (same level as src, results, notebooks, ...) and run the following command:

```sh
python src/data/preprocess.py
```

This will start the preprocessing process, which will read the raw data, perform necessary transformations, and save the preprocessed data to the specified paths in the `config.yml` file.
