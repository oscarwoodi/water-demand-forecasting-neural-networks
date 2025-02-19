# Water Demand Forecasting Preprocessing Guide

This guide will walk you through the steps to set up your environment, configure the necessary files, and run the preprocessing script for the water demand forecasting project. This guide assumes you are new to coding, so each step is explained in detail.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- Python (version 3.6 or higher)
- Git (for cloning the repository)

___
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

Remember that you will also need to activate this environment in any jupyter notebook used also.
___
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
  LOG_DIR: 'results/logs/'

  PREDICTIONS: './results/predictions/'
```

Ensure the paths match the structure of your project directory. These can likely be left alone. 

The following parameters must also be edited to set the type of decompositions performed when preparing the data. Weather feats to be included are added, the parameters for different decomposition types, and specific feature types for the data used such as diurnal flow. 

```yaml
DATA:
  FREQUENCY: ['H']
  PREPROCESS: ['IMPUTE', 'ANOMOLY_REMOVAL', 'DWT', 'CEEMDAN']  # One or multiple of ['IMPUTE', 'CEEMDAN', 'DWT', 'ANOMOLY_REMOVAL']
  CEEMDAN_DECOMPOSITIONS: 5 # None
  DWT_DECOMPOSITIONS: 1 # None
  WEATHER_FEATS: ['AIR_TEMP', 'AIR_HUMIDITY']  # One of ['AIR_TEMP', 'AIR_HUMIDITY', WIND_SPEED', RAIN_DEPTH']
  TIME_FEATS: []   # One of ['WEEKEND', 'HOLIDAY', 'WEEKDAY', 'HOUR']
  SPECIFIC_FEATS: ['DIURNAL']  # One or multiple of ['DIURNAL', 'K_MEANS', 'RESIDUALS']
  START_TRIM: 2000
  END_TRIM: 168
  DMAS: ['dma_a', 'dma_b', 'dma_c', 'dma_d', 'dma_e', 'dma_f', 'dma_g', 'dma_h', 'dma_i', 'dma_j']
```

The main vairables to edit are,
1. **Preprocess** - This applies processes such as imputation and anomaly removal to the raw data during processing. It also allows the option to partition out further datasets based on decomposition methods.
2. **Weather/Time Feats** - For any addittional data that may be used in the models as exogenous variables.
3. **Specific Feats** - These are methods of altering the data so that it can be modelled as e.g. diurnal residuals (See the paper for examples of how this is done)
4. **Start/End Trim** - Start trim may be useful where there is lots of missing data at the start which may not be used in training and end trim if you want to keep some data separate which can be used in further testing e.g. based on the best selected model. 
5. **DMAs** - Put the different DMAs which data should be processed for. 
___
## Step 5: Run the Preprocessing Script

Before running any models, the data should be processed. Please see data > info for examples of what the raw data should look like. 

With the configuration file set up, you can now run the preprocessing script. In your terminal, navigate to the top of the directory (same level as src, results, notebooks, ...) and run the following command:

```sh
python src/data/preprocess.py
```

This will start the preprocessing process, which will read the raw data, perform necessary transformations, and save the preprocessed data to the specified paths in the `config.yml` file.
___
## Step 6: Train and Test Models

Once data has been preprocessed successfully (data > preprocessed > preprocessed.csv) models can be trained by altering the following config arguments. 

```yaml
# Training experiments
TRAIN:
  MODEL: 'gru' # One of ['ann', 'lstm', 'gru', '1dcnn', 'arima', 'sarimax', 'random_forest', 'linear_regression']
  DECOMPOSITION: ['DIURNAL'] # One of ['DIURNAL', 'RESIDUALS', 'DWT', 'CEEMDAN']
  EXPERIMENT: 'cross_validation'        # One of ['train_single', 'train_all', 'hparam_search', 'cross_validation', 'multi_cross_validation']
  VALIDATIONS: 1
  N_QUANTILES: 62
  N_FOLDS: 7
  HPARAM_SEARCH:
    N_EVALS: 40
    HPARAM_OBJECTIVE: 'MSE'             # One of ['MAPE', 'MAE', 'MSE', 'RMSE']
    LAST_FOLDS: 4
  INTERPRETABILITY: true

# Going back to the 'data' input field
DATA: 
  DMAS: ['dma_a', 'dma_b', 'dma_c', 'dma_d', 'dma_e', 'dma_f', 'dma_g', 'dma_h', 'dma_i', 'dma_j']
```

The main variables to edit are,
1. **Model** - A list of the avaialble is give but there may be more not yet listed here. To edit models individually see src > models > ...
2. **Decomposition** - Gives the type of decomposition applied to the data before it is modelled, e.g. Dirunal flow decomposition method can be seen in the paper.
3. **Experiment** - Used for the type of train used, methods such as cross validation and multi-cross validation can be seen in the paper. 

*(For the next parameters it is best to see and understand the logic in train.py to check these descriptions are correct)*

4. **Validations** - In the multi-cross-validation this specifies the number of times that N_FOLD cross validation is done. E.g. for N_FOLD = 7 and VALIDATIONS = 2, 2 weeks of data are tested in total. 
5. **N_Quantiles** - (I believe) This is the total number of days to be included for all training and testing in total. E.g. for n_folds = 7 (so 7 days of iteratively forecasting one day ahead), with 4 weeks of training data and VALIDATIONS = 1, Quantiles should be equal to: 4 weeks train x 7 days + 7 days predicting - 1 (im not sure why -1) = 34. If VALIDATIONS = 2: 4 weeks x 7 days + 14 days predicting - 1 = 41.
6. **N_Folds** - Number of days of iteratively training and then forecasting a day ahead to be done in a single cross validation. 7 = a week of daily forecasts. 
7. **DMAs** - Change the DMAs selected under Data > DMAs to alter which DMAs the model is run on post data processing. 

The results should save under *results > experiments* along with the config used for each result. Visualisations of the forecasts can be seen under *img > forecast_visualizations*. 

See also the notebooks for step by step use of different models, imputation, anomaly removal, stat analysis on data, ...

___

*Thank you for looking at the repo, it is far from perfect to please do look at the code if something doesn't make sense, your hunch is probably right! - Oscar*