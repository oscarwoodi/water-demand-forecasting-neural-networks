import pandas as pd
import yaml
import os
import datetime
import time as time
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from models.prophet import ProphetModel
from models.arima import ARIMAModel
from models.sarimax import SARIMAXModel
from models.nn import *
from models.skmodels import *
from data.preprocess import preprocess_ts
from visualization.visualize import plot_bayesian_hparam_opt

# Map model names to their respective class definitions
MODELS_DEFS = {
    'PROPHET': ProphetModel,
    'ARIMA': ARIMAModel,
    'SARIMAX': SARIMAXModel,
    'LSTM': LSTMModel,
    'GRU': GRUModel,
    '1DCNN': CNN1DModel,
    'LINEAR_REGRESSION': LinearRegressionModel,
    'RANDOM_FOREST': RandomForestModel ,
    'ANN': ANNModel
}


def load_dataset(cfg, fixed_test_set=True):
    '''
    Load preprocessed dataset and return training and test sets.
    :param cfg: Project config
    :return: DataFrames for training and test sets
    '''
    try:
        if 'DWT' in cfg['TRAIN']['DECOMPOSITION']: 
            df = pd.read_csv(f"{cfg['PATHS']['PREPROCESSED_DWT_DATA'][:-4]}_{cfg['TRAIN']['DECOMPOSITION'][-1].lower()}.csv")
        elif 'CEEMDAN' in cfg['TRAIN']['DECOMPOSITION']: 
            df = pd.read_csv(f"{cfg['PATHS']['PREPROCESSED_CEEMDAN_DATA'][:-4]}_{cfg['TRAIN']['DECOMPOSITION'][-1].lower()}.csv")
        else: 
            df = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
        
        if cfg['DATA']['CURRENT_DMA'] != None:
            df['Consumption'] = df[cfg['DATA']['CURRENT_DMA']]
            if 'DIURNAL' in cfg['DATA']['SPECIFIC_FEATS']: 
                df['Diurnal'] = df[['diurnal_'+cfg['DATA']['CURRENT_DMA']]]
            cols = [col for col in df.columns if 'dma' not in col.split('_')]
            df = df[cols]

    except FileNotFoundError:
        print("No file found at " + cfg['PATHS']['PREPROCESSED_DATA'] + ". Running preprocessing of client data.")
        df = preprocess_ts(cfg, save_raw_df=True, save_prepr_df=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Define training and test sets
    if fixed_test_set:
        if cfg['DATA']['TEST_DAYS'] <= 0:
            train_df = df[:int(df.shape[0])]
            test_df = df[int(df.shape[0]):]
        else:
            df = df[int(-cfg['DATA']['TRAIN_DAYS']-cfg['DATA']['TEST_DAYS']-cfg['DATA']['END_TRIM']):int(-cfg['DATA']['END_TRIM'])].reset_index(drop=True)
            train_df = df[int(-cfg['DATA']['TRAIN_DAYS']-cfg['DATA']['TEST_DAYS']):int(-cfg['DATA']['TEST_DAYS'])]
            test_df = df[int(-cfg['DATA']['TEST_DAYS']):]
    else:
        train_df = df[:int((1 - cfg['DATA']['TEST_FRAC']) * df.shape[0])]
        test_df = df[int((1 - cfg['DATA']['TEST_FRAC']) * df.shape[0]):]
    print('Size of training set: ', train_df.shape[0])
    print('Size of test set: ', test_df.shape[0])

    return train_df, test_df, df


def train_model(cfg, model_def, hparams, train_df, test_df, save_model=False, write_logs=False, save_metrics=False, dated_paths=True):
    '''
    Train a model
    :param cfg: Project config
    :param model_def: Class definition of model to train
    :param hparams: A dict of hyperparameters specific to this model
    :param train_df: Training set as DataFrame
    :param test_df: Test set as DataFrame
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :param save_metrics: Flag indicating whether to save the forecast metrics to a CSV
    :param dated_paths: Flag indicating whether to include train date in outputs paths
    :return: Dictionary of test set forecast metrics
    '''
    log_dir = cfg['PATHS']['LOGS'] if write_logs else None
    model = model_def(hparams, log_dir=log_dir)  # Create instance of model
    if not dated_paths:
        model.train_date = ''

    # Fit the model
    if model.univariate:
        train_df = train_df[['Date', 'Consumption']]
        test_df = test_df[['Date', 'Consumption']]

    model.fit(train_df)

    if save_model:
        model.save(cfg['PATHS']['MODELS'], scaler_dir=cfg['PATHS']['SERIALIZATIONS'])

    # Evaluate the model on the test set, if it exists
    if test_df.shape[0] > 0:
        save_dir = cfg['PATHS']['EXPERIMENTS'] if save_metrics else None
        test_forecast_metrics, forecast_df = model.evaluate(train_df, test_df, save_dir=save_dir, plot=save_metrics)
    else:
        test_forecast_metrics = {}
        
    # If we are training a Prophet model, decompose it and save the components' parameters and visualization
    if cfg['TRAIN']['INTERPRETABILITY'] and model.name == 'Prophet':
        model.decompose(cfg['PATHS']['INTERPRETABILITY'], cfg['PATHS']['INTERPRETABILITY_VISUALIZATIONS'])
    return test_forecast_metrics, forecast_df, model



def train_single(cfg, hparams=None, save_model=False, write_logs=False, save_metrics=False, fixed_test_set=False, dated_paths=True):
    '''
    Train a single model. Use the passed hyperparameters if possible; otherwise, use those in config.
    :param cfg: Project config
    :param hparams: Dict of hyperparameters
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :param save_metrics: Flag indicating whether to save the forecast metrics to a CSV
    :param fixed_test_set: Flag indicating whether to use a fixed number of days for test set
    :param dated_paths: Whether to include current datetime in persistent storage paths
    :return: Dictionary of test set forecast metrics
    '''
    train_df, test_df = load_dataset(cfg, fixed_test_set=fixed_test_set)
    model_def = MODELS_DEFS.get(cfg['TRAIN']['MODEL'].upper(), lambda: "Invalid model specified in cfg['TRAIN']['MODEL']")
    if hparams is None:
        hparams = cfg['HPARAMS'][cfg['TRAIN']['MODEL'].upper()]
    test_forecast_metrics, forecast_df, model = train_model(cfg, model_def, hparams, train_df, test_df, save_model=save_model,
                                        write_logs=write_logs, save_metrics=save_metrics, dated_paths=dated_paths)
    print('Test forecast metrics: ', test_forecast_metrics)
    return test_forecast_metrics, forecast_df, model


def train_all(cfg, save_models=False, write_logs=False):
    '''
    Train all models that have available definitions in this project
    :param cfg: Project config
    :param save_models: Flag indicating whether to save the trained models
    :param write_logs: Flag indicating whether to write any training logs to disk
    :return: DataFrame of test set forecast metrics for all models
    '''
    train_df, test_df = load_dataset(cfg)
    all_model_metrics = {}
    for model_name in MODELS_DEFS:
        print('*** Training ' + model_name + ' ***\n')
        model_def = MODELS_DEFS[model_name]
        hparams = cfg['HPARAMS'][model_name]
        test_forecast_metrics, forecast_df, _ = train_model(cfg, model_def, hparams, train_df, test_df, save_model=save_models,
                                            write_logs=write_logs)
        if all_model_metrics:
            all_model_metrics['model'].append(model_name)
            for metric in test_forecast_metrics:
                all_model_metrics[metric].append(test_forecast_metrics[metric])
        else:
            all_model_metrics['model'] = [model_name]
            for metric in test_forecast_metrics:
                all_model_metrics[metric] = [test_forecast_metrics[metric]]
        print('Test forecast metrics for ' + model_name + ': ', test_forecast_metrics)
    metrics_df = pd.DataFrame(all_model_metrics)
    file_name = 'all_train' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv'
    metrics_df.to_csv(os.path.join(cfg['PATHS']['EXPERIMENTS'], file_name), columns=metrics_df.columns,
                      index_label=False, index=False)
    return metrics_df, forecast_df


def cross_validation(cfg, dataset=None, metrics=None, model_name=None, hparams=None, save_results=False, fixed_test_set=True, save_individual=False):
    '''
    Perform a nested cross-validation with day-forward chaining. Results are saved in CSV format.
    :param cfg: project config
    :param dataset: A DataFrame consisting of the entire dataset
    :param metrics: list of metrics to report
    :param model_name: String identifying model
    :param save_results: Flag indicating whether to save results
    :return DataFrame of metrics
    '''

    n_quantiles = cfg['TRAIN']['N_QUANTILES']
    n_folds = cfg['TRAIN']['N_FOLDS']
    if dataset is None:
        _, _, dataset = load_dataset(cfg, fixed_test_set=fixed_test_set)
    if n_folds is None:
        n_folds = n_quantiles
    if metrics is None:
        metrics = ['residuals_mean', 'residuals_std', 'error_mean', 'error_std', 'MAE', 'MAPE', 'MSE', 'RMSE']
    n_rows = n_quantiles if n_folds is None else n_folds
    metrics_df = pd.DataFrame(np.zeros((n_rows + 2, len(metrics) + 1)), columns=['Fold'] + metrics)
    metrics_df['Fold'] = list(range(n_quantiles - n_folds + 1, n_quantiles + 1)) + ['mean', 'std']
    ts_cv = TimeSeriesSplit(n_splits=n_quantiles)
    model_name = cfg['TRAIN']['MODEL'].upper() if model_name is None else model_name
    hparams = cfg['HPARAMS'][model_name] if hparams is None else hparams

    model_def = MODELS_DEFS.get(model_name, lambda: "Invalid model specified in cfg['TRAIN']['MODEL']")

    # save model params
    if save_individual: 
        save_dir = f"{cfg['PATHS']['EXPERIMENTS']}/{cfg['DATA']['SAVE_LABEL']}/" if save_results else None
        plot = True
    else: 
        save_dir = None
        plot = False

    # create results directory

    # Train a model n_folds times with different folds
    cur_fold = 0
    row_idx = 0
    for train_index, test_index in ts_cv.split(dataset):
        if cur_fold >= n_quantiles - n_folds:
            print('Fitting model for fold ' + str(cur_fold))
            model = model_def(hparams)
            model.train_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

            # Separate into training and test sets
            train_df, test_df = dataset.iloc[train_index], dataset.iloc[test_index]
            if model.univariate:
                train_df = train_df[['Date', 'Consumption']]
                test_df = test_df[['Date', 'Consumption']]

            # Train the model and evaluate performance on test set
            model.fit(train_df)
            
            test_metrics, forecast_df = model.evaluate(train_df, test_df, save_dir=save_dir, plot=plot)
            for metric in test_metrics:
                if metric in metrics_df.columns:
                    metrics_df.loc[row_idx, metric] = test_metrics[metric]
            row_idx += 1
        cur_fold += 1

    # Record mean and standard deviation of test set results
    for metric in metrics:
        metrics_df.loc[n_folds, metric] = metrics_df[metric][0:-2].mean()
        metrics_df.loc[n_folds + 1, metric] = metrics_df[metric][0:-2].std()

    # Save results
    if save_results:
        file_path = f"{cfg['PATHS']['EXPERIMENTS']}/{cfg['DATA']['SAVE_LABEL']}/" + 'cross_val_' + model_name + \
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
        metrics_df.to_csv(file_path, columns=metrics_df.columns, index_label=False, index=False)
        

    print(metrics_df)

    return metrics_df, forecast_df, model


def bayesian_hparam_optimization(cfg):
    '''
    Conducts a Bayesian hyperparameter optimization, given the parameter ranges and selected model
    :param cfg: Project config
    :return: Dict of hyperparameters deemed optimal
    '''

    #dataset = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
    #dataset['Date'] = pd.to_datetime(dataset['Date'])

    model_name = cfg['TRAIN']['MODEL'].upper()
    objective_metric = cfg['TRAIN']['HPARAM_SEARCH']['HPARAM_OBJECTIVE']
    results = {'Trial': [], objective_metric: []}
    dimensions = []
    default_params = []
    hparam_names = []
    for hparam_name in cfg['HPARAM_SEARCH'][model_name]:
        if cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'] is not None:
            if cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'set':
                dimensions.append(Categorical(categories=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'],
                                              name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'int_uniform':
                dimensions.append(Integer(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                          high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                          prior='uniform', name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_log':
                dimensions.append(Real(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                       high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                       prior='log-uniform', name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_uniform':
                dimensions.append(Real(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                       high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                       prior='uniform', name=hparam_name))
            default_params.append(cfg['HPARAMS'][model_name][hparam_name])
            hparam_names.append(hparam_name)
            results[hparam_name] = []

    def objective(vals):
        hparams = dict(zip(hparam_names, vals))
        print('HPARAM VALUES: ', hparams)
        scores, _, _ = cross_validation(cfg, metrics=[objective_metric], model_name=model_name, hparams=hparams, save_results=False, save_individual=False)
        scores = scores[objective_metric]
        score = scores[scores.shape[0] - 2]     # Get the mean value for the error metric from the cross validation
        #test_metrics, _ = train_single(cfg, hparams=hparams, save_model=False, write_logs=False, save_metrics=False)
        #score = test_metrics['MAPE']
        return score   # We aim to minimize error
    search_results = gp_minimize(func=objective, dimensions=dimensions, acq_func='EI',
                                 n_calls=cfg['TRAIN']['HPARAM_SEARCH']['N_EVALS'], verbose=True)
    print(search_results)
    #plot_bayesian_hparam_opt(model_name, hparam_names, search_results, save_fig=True)

    # Create table to detail results
    trial_idx = 0
    for t in search_results.x_iters:
        results['Trial'].append(str(trial_idx))
        results[objective_metric].append(search_results.func_vals[trial_idx])
        for i in range(len(hparam_names)):
            results[hparam_names[i]].append(t[i])
        trial_idx += 1
    results['Trial'].append('Best')
    results[objective_metric].append(search_results.fun)
    for i in range(len(hparam_names)):
        results[hparam_names[i]].append(search_results.x[i])
    results_df = pd.DataFrame(results)
    if cfg['TRAIN']['DECOMPOSITION'] == []:
        results_path = cfg['PATHS']['EXPERIMENTS']+'/'+cfg['DATA']['SAVE_LABEL']+'/'+cfg['DATA']['CURRENT_DMA']+'_hparam_search_'+model_name+\
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
    else: 
        results_path = cfg['PATHS']['EXPERIMENTS']+'/'+cfg['DATA']['SAVE_LABEL']+'/'+cfg['DATA']['CURRENT_DMA']+'_'+cfg['TRAIN']['DECOMPOSITION'][-1]+'_hparam_search_'+model_name+\
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
        
    results_df.to_csv(results_path, index_label=False, index=False)


    return search_results


def multi_train(cfg, experiment, save_model):
    '''
    Deals with instances where multiple models require training at once to be combined.
    :param cfg: project config
    '''
    # number of model trained
    if 'CEEMDAN' in cfg['TRAIN']['DECOMPOSITION']: 
        no_trains = cfg['DATA']['CEEMDAN_DECOMPOSITIONS']
        # alteration to change dataset used to residuals of dwt
        cfg['TRAIN']['DECOMPOSITION'].append('CEEMDAN_0')
    elif 'DWT' in cfg['TRAIN']['DECOMPOSITION']: 
        no_trains = 2
        # alteration to change dataset used to residuals of dwt
        cfg['TRAIN']['DECOMPOSITION'].append('DWT_APPROX')

    for i in range(no_trains): 
        # label of data for train
        data_label = cfg['TRAIN']['DECOMPOSITION'][-1]

        # Conduct the desired train experiment
        if experiment == 'train_single':
            _, forecast_df, model = train_single(cfg, save_model=save_model, save_metrics=True, fixed_test_set=True)
        elif experiment == 'train_all':
            _, forecast_df = train_all(cfg, save_models=save_model)
        elif experiment == 'hparam_search':
            bayesian_hparam_optimization(cfg)
        elif experiment == 'cross_validation':
            _, forecast_df, model = cross_validation(cfg, save_results=True, fixed_test_set=True, save_individual=True)
        else:
            raise Exception("Invalid entry in TRAIN > EXPERIMENT field of config.yml.")

        # update series to fit model for
        if 'CEEMDAN' in cfg['TRAIN']['DECOMPOSITION']: 
            # alteration to change dataset used to residuals of dwt
            cfg['TRAIN']['DECOMPOSITION'].append(f"CEEMDAN_{int(data_label[-1])+1}")
        elif 'DWT' in cfg['TRAIN']['DECOMPOSITION']: 
            # alteration to change dataset used to residuals of dwt
            cfg['TRAIN']['DECOMPOSITION'].append('DWT_RESIDUAL')

        # concatenate results
        if experiment != 'hparam_search':
            if i == 0: 
                total_forecast_df = forecast_df
            else: 
                total_forecast_df = total_forecast_df + forecast_df

            # get results and save
            save_dir = cfg['PATHS']['EXPERIMENTS']
            model.name += '_combined'
            test_metrics, forecast_df = model.evaluate_forecast(total_forecast_df, save_dir=save_dir, plot=True)
            print('\n\nPrinting combined model residuals...')
            print('Test forecast metrics: ', test_metrics)

    return


def train_experiment(cfg=None, experiment='single_train', save_model=False, write_logs=False):
    '''
    Run a training experiment
    :param cfg: Project config
    :param experiment: String defining which experiment to run
    :param save_model: Flag indicating whether to save any models trained during the experiment
    :param write_logs: Flag indicating whether to write logs for training
    '''

    # Load project config data
    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    try: 
        os.mkdir(f"{cfg['PATHS']['EXPERIMENTS']}/{cfg['DATA']['SAVE_LABEL']}")
    except: 
        None

    # Conduct the desired train experiment
    if cfg['TRAIN']['DECOMPOSITION'] != []:
        multi_train(cfg, experiment, save_model)
    elif experiment == 'train_single':
        train_single(cfg, save_model=save_model, save_metrics=True, fixed_test_set=True)
    elif experiment == 'train_all':
        train_all(cfg, save_models=save_model)
    elif experiment == 'hparam_search':
        bayesian_hparam_optimization(cfg)
    elif experiment == 'cross_validation':
        cross_validation(cfg, save_results=True, fixed_test_set=True, save_individual=True)

    else:
        raise Exception("Invalid entry in TRAIN > EXPERIMENT field of config.yml.")
    
    # save config to experiment folder
    with open(f"{cfg['PATHS']['EXPERIMENTS']}/{cfg['DATA']['SAVE_LABEL']}/{cfg['DATA']['CURRENT_DMA']}_{experiment}_config.yml", 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)
    
    return


if __name__ == '__main__':
    cfg = yaml.full_load(open("./config.yml", 'r'))
    start = time.time()
    # run for each dma
    if cfg['DATA']['DMAS'] != None: 
        for dma in cfg['DATA']['DMAS']: 
            print('Running for DMA '+dma)
            cfg['DATA']['CURRENT_DMA'] = dma
            train_experiment(cfg=cfg, experiment=cfg['TRAIN']['EXPERIMENT'], save_model=True, write_logs=True)
    else: 
        train_experiment(cfg=cfg, experiment=cfg['TRAIN']['EXPERIMENT'], save_model=True, write_logs=True)
    end = time.time() - start
    print(end)
