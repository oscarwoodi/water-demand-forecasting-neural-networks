import pandas as pd
import yaml
import os
import datetime
import time as time
import logging
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from models.arima import ARIMAModel
from models.sarimax import SARIMAXModel
from models.nn import *
from data.preprocess import preprocess_ts
from visualization.visualize import plot_bayesian_hparam_opt

# from models.prophet import ProphetModel
# from models.skmodels import *

# Map model names to their respective class definitions
MODELS_DEFS = {
    # 'PROPHET': ProphetModel,
    # 'LINEAR_REGRESSION': LinearRegressionModel,
    # 'RANDOM_FOREST': RandomForestModel,
    "ARIMA": ARIMAModel,
    "SARIMAX": SARIMAXModel,
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "1DCNN": CNN1DModel,
    "ANN": ANNModel,
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_dataset(cfg, fixed_test_set=True):
    """
    Load preprocessed dataset and return training and test sets.
    :param cfg: Project config
    :return: DataFrames for training and test sets
    """
    time_feats = cfg["DATA"]["TIME_FEATS"]
    weather_feats = cfg["DATA"]["WEATHER_FEATS"]
    features = time_feats + weather_feats

    try:
        # get decomposed data if required
        if "DWT" in cfg["TRAIN"]["DECOMPOSITION"]:
            path = f"{cfg['PATHS']['PREPROCESSED_DWT_DATA'][:-4]}_{cfg['TRAIN']['DECOMPOSITION'][-1].lower()}"
            df = pd.read_csv(f"{path}.csv")
        elif "CEEMDAN" in cfg["TRAIN"]["DECOMPOSITION"]:
            path = f"{cfg['PATHS']['PREPROCESSED_CEEMDAN_DATA'][:-4]}_{cfg['TRAIN']['DECOMPOSITION'][-1].lower()}"
            df = pd.read_csv(f"{path}.csv")
        else:
            path = cfg["PATHS"]["PREPROCESSED_DATA"]
            df = pd.read_csv(path)

        # separate out individual dma
        if cfg["DATA"]["CURRENT_DMA"] != None:
            df["Consumption"] = df[cfg["DATA"]["CURRENT_DMA"]]
            # add diurnal data if required
            if "DIURNAL" in cfg["TRAIN"]["DECOMPOSITION"]:
                df["Diurnal"] = df[["diurnal_" + cfg["DATA"]["CURRENT_DMA"]]]
                features += ["Diurnal"]

            # add sarimax residual to dataset
            if "RESIDUALS" in cfg["TRAIN"]["DECOMPOSITION"]:
                path = cfg["PATHS"]["PREPROCESSED_MODEL_RESIDUAL_DATA"]
                sarimax_result = pd.read_csv(path)
                residual_model = (
                    sarimax_result["Consumption_" + cfg["DATA"]["CURRENT_DMA"]]
                    .fillna(0)
                    .astype("float64")
                    .tolist()
                )
                forecast_model = (
                    sarimax_result["Forecast_" + cfg["DATA"]["CURRENT_DMA"]]
                    .fillna(0)
                    .astype("float64")
                    .tolist()
                )
                df["Residual"] = 0.0
                df["Residual_forecast"] = 0.0
                df["Residual"].iloc[-int(len(residual_model)) :] = residual_model
                df["Residual_forecast"].iloc[
                    -int(len(forecast_model)) :
                ] = forecast_model
                features += ["Residual"]
                features += ["Residual_forecast"]

            df = df.loc[:, ["Date", "Consumption"] + features]
            cols = [col for col in df.columns if "dma" not in col.split("_")]
            df = df[cols]

    except FileNotFoundError:
        logging.error(
            "No file found at %s. Running preprocessing of client data.", path
        )
        df = preprocess_ts(cfg, save_raw_df=True, save_prepr_df=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # Define training and test sets
    if fixed_test_set:
        if cfg["DATA"]["TEST_DAYS"] <= 0:
            train_df = df[: int(df.shape[0])]
            test_df = df[int(df.shape[0]) :]
        else:
            df = df[
                int(-cfg["DATA"]["TRAIN_DAYS"] - cfg["DATA"]["TEST_DAYS"]) :
            ].reset_index(drop=True)
            train_df = df[
                int(-cfg["DATA"]["TRAIN_DAYS"] - cfg["DATA"]["TEST_DAYS"]) : int(
                    -cfg["DATA"]["TEST_DAYS"]
                )
            ]
            test_df = df[int(-cfg["DATA"]["TEST_DAYS"]) :]
    else:
        train_df = df[: int((1 - cfg["DATA"]["TEST_FRAC"]) * df.shape[0])]
        test_df = df[int((1 - cfg["DATA"]["TEST_FRAC"]) * df.shape[0]) :]
    logging.info("Size of training set: %d", train_df.shape[0])
    logging.info("Size of test set: %d", test_df.shape[0])

    return train_df, test_df, df


def train_model(
    cfg,
    model_def,
    hparams,
    train_df,
    test_df,
    save_model=False,
    write_logs=False,
    save_metrics=False,
    dated_paths=True,
):
    """
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
    """
    log_dir = cfg["PATHS"]["LOGS"] if write_logs else None
    model = model_def(hparams, log_dir=log_dir)  # Create instance of model
    if not dated_paths:
        model.train_date = ""

    model.fit(train_df)

    if save_model:
        model.save(cfg["PATHS"]["MODELS"], scaler_dir=cfg["PATHS"]["SERIALIZATIONS"])

    # Evaluate the model on the test set, if it exists
    if test_df.shape[0] > 0:
        save_dir = cfg["PATHS"]["EXPERIMENTS"] if save_metrics else None
        test_forecast_metrics, forecast_df = model.evaluate(
            train_df, test_df, save_dir=save_dir, plot=save_metrics
        )
    else:
        test_forecast_metrics = {}

    # If we are training a Prophet model, decompose it and save the components' parameters and visualization
    if cfg["TRAIN"]["INTERPRETABILITY"] and model.name == "Prophet":
        model.decompose(
            cfg["PATHS"]["INTERPRETABILITY"],
            cfg["PATHS"]["INTERPRETABILITY_VISUALIZATIONS"],
        )
    return test_forecast_metrics, forecast_df, model


def train_single(
    cfg,
    hparams=None,
    save_model=False,
    write_logs=False,
    save_metrics=False,
    fixed_test_set=False,
    dated_paths=True,
):
    """
    Train a single model. Use the passed hyperparameters if possible; otherwise, use those in config.
    :param cfg: Project config
    :param hparams: Dict of hyperparameters
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :param save_metrics: Flag indicating whether to save the forecast metrics to a CSV
    :param fixed_test_set: Flag indicating whether to use a fixed number of days for test set
    :param dated_paths: Whether to include current datetime in persistent storage paths
    :return: Dictionary of test set forecast metrics
    """
    train_df, test_df = load_dataset(cfg, fixed_test_set=fixed_test_set)
    model_def = MODELS_DEFS.get(
        cfg["TRAIN"]["MODEL"].upper(),
        lambda: "Invalid model specified in cfg['TRAIN']['MODEL']",
    )
    if hparams is None:
        hparams = cfg["HPARAMS"][cfg["TRAIN"]["MODEL"].upper()]
    test_forecast_metrics, forecast_df, model = train_model(
        cfg,
        model_def,
        hparams,
        train_df,
        test_df,
        save_model=save_model,
        write_logs=write_logs,
        save_metrics=save_metrics,
        dated_paths=dated_paths,
    )
    logging.info("Test forecast metrics: %s", test_forecast_metrics)
    return test_forecast_metrics, forecast_df, model


def train_all(cfg, save_models=False, write_logs=False):
    """
    Train all models that have available definitions in this project
    :param cfg: Project config
    :param save_models: Flag indicating whether to save the trained models
    :param write_logs: Flag indicating whether to write any training logs to disk
    :return: DataFrame of test set forecast metrics for all models
    """
    train_df, test_df = load_dataset(cfg)
    all_model_metrics = {}
    for model_name in MODELS_DEFS:
        logging.info("*** Training %s ***", model_name)
        model_def = MODELS_DEFS[model_name]
        hparams = cfg["HPARAMS"][model_name]
        test_forecast_metrics, forecast_df, _ = train_model(
            cfg,
            model_def,
            hparams,
            train_df,
            test_df,
            save_model=save_models,
            write_logs=write_logs,
        )
        if all_model_metrics:
            all_model_metrics["model"].append(model_name)
            for metric in test_forecast_metrics:
                all_model_metrics[metric].append(test_forecast_metrics[metric])
        else:
            all_model_metrics["model"] = [model_name]
            for metric in test_forecast_metrics:
                all_model_metrics[metric] = [test_forecast_metrics[metric]]
        logging.info(
            "Test forecast metrics for %s: %s", model_name, test_forecast_metrics
        )
    metrics_df = pd.DataFrame(all_model_metrics)
    file_name = "all_train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    metrics_df.to_csv(
        os.path.join(cfg["PATHS"]["EXPERIMENTS"], file_name),
        columns=metrics_df.columns,
        index_label=False,
        index=False,
    )
    return metrics_df, forecast_df


def cross_validation(
    cfg,
    dataset=None,
    metrics=None,
    model_name=None,
    hparams=None,
    save_results=False,
    fixed_test_set=True,
    save_individual=False,
):
    """
    Perform a nested cross-validation with day-forward chaining. Results are saved in CSV format.
    :param cfg: project config
    :param dataset: A DataFrame consisting of the entire dataset
    :param metrics: list of metrics to report
    :param model_name: String identifying model
    :param save_results: Flag indicating whether to save results
    :return DataFrame of metrics
    """
    n_quantiles = cfg["TRAIN"]["N_QUANTILES"]
    n_folds = cfg["TRAIN"]["N_FOLDS"]
    if dataset is None:
        _, _, dataset = load_dataset(cfg, fixed_test_set=fixed_test_set)
    if n_folds is None:
        n_folds = n_quantiles
    if metrics is None:
        metrics = [
            "residuals_mean",
            "residuals_std",
            "error_mean",
            "error_std",
            "MAE",
            "MAPE",
            "MSE",
            "RMSE",
        ]
    n_rows = n_quantiles if n_folds is None else n_folds
    metrics_df = pd.DataFrame(
        np.zeros((n_rows + 2, len(metrics) + 1)), columns=["Fold"] + metrics
    )
    metrics_df["Fold"] = list(range(n_quantiles - n_folds + 1, n_quantiles + 1)) + [
        "mean",
        "std",
    ]
    forecast_df = pd.DataFrame()
    ts_cv = TimeSeriesSplit(n_splits=n_quantiles)
    model_name = cfg["TRAIN"]["MODEL"].upper() if model_name is None else model_name
    hparams = cfg["HPARAMS"][model_name] if hparams is None else hparams

    model_def = MODELS_DEFS.get(
        model_name, lambda: "Invalid model specified in cfg['TRAIN']['MODEL']"
    )

    # save model params
    if save_individual:
        save_dir = (
            f"{cfg['PATHS']['EXPERIMENTS']}/{cfg['DATA']['SAVE_LABEL']}/"
            if save_results
            else None
        )
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
            logging.info("Fitting model for fold %d", cur_fold)
            model = model_def(hparams)
            model.train_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            # Separate into training and test sets
            train_df, test_df = dataset.iloc[train_index], dataset.iloc[test_index]
            # Train the model and evaluate performance on test set
            model.fit(train_df)

            test_metrics, test_forecast_df = model.evaluate(
                train_df, test_df, save_dir=save_dir, plot=plot
            )
            for metric in test_metrics:
                if metric in metrics_df.columns:
                    metrics_df.loc[row_idx, metric] = test_metrics[metric]
            forecast_df = pd.concat(
                [
                    forecast_df,
                    test_forecast_df.rename(
                        columns={
                            col: col + "_fold_" + str(row_idx)
                            for col in test_forecast_df.columns
                        }
                    ),
                ],
                axis=1,
            )
            row_idx += 1
        cur_fold += 1

    # Record mean and standard deviation of test set results
    for metric in metrics:
        metrics_df.loc[n_folds, metric] = metrics_df[metric][0:-2].mean()
        metrics_df.loc[n_folds + 1, metric] = metrics_df[metric][0:-2].std()

    # Save results
    if save_results:
        file_path = (
            f"{cfg['PATHS']['EXPERIMENTS']}/{cfg['DATA']['SAVE_LABEL']}/"
            + "cross_val_"
            + model_name
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            + ".csv"
        )
        metrics_df.to_csv(
            file_path, columns=metrics_df.columns, index_label=False, index=False
        )

    return metrics_df, forecast_df, model


def bayesian_hparam_optimization(cfg):
    """
    Conducts a Bayesian hyperparameter optimization, given the parameter ranges and selected model
    :param cfg: Project config
    :return: Dict of hyperparameters deemed optimal
    """

    # dataset = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
    # dataset['Date'] = pd.to_datetime(dataset['Date'])

    model_name = cfg["TRAIN"]["MODEL"].upper()
    objective_metric = cfg["TRAIN"]["HPARAM_SEARCH"]["HPARAM_OBJECTIVE"]
    results = {"Trial": [], objective_metric: []}
    dimensions = []
    default_params = []
    hparam_names = []
    for hparam_name in cfg["HPARAM_SEARCH"][model_name]:
        if cfg["HPARAM_SEARCH"][model_name][hparam_name]["RANGE"] is not None:
            if cfg["HPARAM_SEARCH"][model_name][hparam_name]["TYPE"] == "set":
                dimensions.append(
                    Categorical(
                        categories=cfg["HPARAM_SEARCH"][model_name][hparam_name][
                            "RANGE"
                        ],
                        name=hparam_name,
                    )
                )
            elif cfg["HPARAM_SEARCH"][model_name][hparam_name]["TYPE"] == "int_uniform":
                dimensions.append(
                    Integer(
                        low=cfg["HPARAM_SEARCH"][model_name][hparam_name]["RANGE"][0],
                        high=cfg["HPARAM_SEARCH"][model_name][hparam_name]["RANGE"][1],
                        prior="uniform",
                        name=hparam_name,
                    )
                )
            elif cfg["HPARAM_SEARCH"][model_name][hparam_name]["TYPE"] == "float_log":
                dimensions.append(
                    Real(
                        low=cfg["HPARAM_SEARCH"][model_name][hparam_name]["RANGE"][0],
                        high=cfg["HPARAM_SEARCH"][model_name][hparam_name]["RANGE"][1],
                        prior="log-uniform",
                        name=hparam_name,
                    )
                )
            elif (
                cfg["HPARAM_SEARCH"][model_name][hparam_name]["TYPE"] == "float_uniform"
            ):
                dimensions.append(
                    Real(
                        low=cfg["HPARAM_SEARCH"][model_name][hparam_name]["RANGE"][0],
                        high=cfg["HPARAM_SEARCH"][model_name][hparam_name]["RANGE"][1],
                        prior="uniform",
                        name=hparam_name,
                    )
                )
            default_params.append(cfg["HPARAMS"][model_name][hparam_name])
            hparam_names.append(hparam_name)
            results[hparam_name] = []

    def objective(vals):
        hparams = dict(zip(hparam_names, vals))
        logging.info("HPARAM VALUES: %s", hparams)
        scores, _, _ = cross_validation(
            cfg,
            metrics=[objective_metric],
            model_name=model_name,
            hparams=hparams,
            save_results=False,
            save_individual=False,
        )
        scores = scores[objective_metric]
        score = scores[
            scores.shape[0] - 2
        ]  # Get the mean value for the error metric from the cross validation
        # test_metrics, _ = train_single(cfg, hparams=hparams, save_model=False, write_logs=False, save_metrics=False)
        # score = test_metrics['MAPE']
        return score  # We aim to minimize error

    search_results = gp_minimize(
        func=objective,
        dimensions=dimensions,
        acq_func="EI",
        n_calls=cfg["TRAIN"]["HPARAM_SEARCH"]["N_EVALS"],
        verbose=True,
    )
    logging.info(search_results)
    # plot_bayesian_hparam_opt(model_name, hparam_names, search_results, save_fig=True)

    # Create table to detail results
    trial_idx = 0
    for t in search_results.x_iters:
        results["Trial"].append(str(trial_idx))
        results[objective_metric].append(search_results.func_vals[trial_idx])
        for i in range(len(hparam_names)):
            results[hparam_names[i]].append(t[i])
        trial_idx += 1
    results["Trial"].append("Best")
    results[objective_metric].append(search_results.fun)
    for i in range(len(hparam_names)):
        results[hparam_names[i]].append(search_results.x[i])
    results_df = pd.DataFrame(results)
    if cfg["TRAIN"]["DECOMPOSITION"] == []:
        results_path = (
            cfg["PATHS"]["EXPERIMENTS"]
            + "/"
            + cfg["DATA"]["SAVE_LABEL"]
            + "/"
            + cfg["DATA"]["CURRENT_DMA"]
            + "_hparam_search_"
            + model_name
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            + ".csv"
        )
    else:
        results_path = (
            cfg["PATHS"]["EXPERIMENTS"]
            + "/"
            + cfg["DATA"]["SAVE_LABEL"]
            + "/"
            + cfg["DATA"]["CURRENT_DMA"]
            + "_"
            + cfg["TRAIN"]["DECOMPOSITION"][-1]
            + "_hparam_search_"
            + model_name
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            + ".csv"
        )

    results_df.to_csv(results_path, index_label=False, index=False)

    return search_results


def multi_cross_validation(
    cfg,
    save_results=True,
    fixed_test_set=True,
    save_individual=False,
    dataset=None,
    metrics=None,
    model_name=None,
    hparams=None,
):
    """
    Trains and cross-validates model multiple times to get distribution of results.
    :param cfg: project config
    :param save_results: Flag indicating whether to save results
    :param save_individual: Flag indicating whether to save each individual fold result and plot
    :return DataFrame of metrics and predictions from each run
    """
    no_validations = cfg["TRAIN"]["VALIDATIONS"]
    model_name = cfg["TRAIN"]["MODEL"].upper() if model_name is None else model_name

    # pre allocate result dictionarys
    combined_metrics = {run: None for run in range(no_validations)}
    combined_forecast = {run: None for run in range(no_validations)}

    for i in range(no_validations):
        logging.info("Running Cross Validation No. %d...", i + 1)
        # run individual cross validations and save metric for each if save_individual=True
        metrics, forecast_df, model = cross_validation(
            cfg,
            save_results=save_individual,
            fixed_test_set=fixed_test_set,
            save_individual=False,
            model_name=model_name,
            hparams=hparams,
            dataset=dataset,
        )
        combined_metrics[i] = metrics
        combined_forecast[i] = forecast_df.rename(
            columns={col: col + "_" + str(i) for col in forecast_df.columns}
        )

    # combine results and get mean metrics
    combined_metrics_df = pd.concat([df for df in combined_metrics.values()], axis=1)
    mean_metrics_df = combined_metrics_df[["Fold"]].copy()
    for metric in [
        metric for metric in combined_metrics_df.columns.unique() if metric != "Fold"
    ]:
        if no_validations > 1:
            mean_metrics_df[metric + "_mean"] = combined_metrics_df[metric].mean(
                axis="columns"
            )
            mean_metrics_df[metric + "_std"] = combined_metrics_df[metric].std(
                axis="columns"
            )
        else:
            mean_metrics_df[metric + "_mean"] = combined_metrics_df[metric]
            mean_metrics_df[metric + "_std"] = combined_metrics_df[metric]

    mean_metrics_df = mean_metrics_df.T.drop_duplicates().T

    combined_forecast_df = pd.concat([df for df in combined_forecast.values()], axis=1)

    # save results
    evalutation_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if save_results:
        if cfg["TRAIN"]["DECOMPOSITION"] == []:
            results_path = (
                cfg["PATHS"]["EXPERIMENTS"]
                + "/"
                + cfg["DATA"]["SAVE_LABEL"]
                + "/"
                + cfg["DATA"]["CURRENT_DMA"]
                + "_multi_validation_"
                + model_name
            )
        else:
            results_path = (
                cfg["PATHS"]["EXPERIMENTS"]
                + "/"
                + cfg["DATA"]["SAVE_LABEL"]
                + "/"
                + cfg["DATA"]["CURRENT_DMA"]
                + "_"
                + cfg["TRAIN"]["DECOMPOSITION"][-1]
                + "_multi_validation_"
                + model_name
            )

        mean_metrics_df.to_csv(
            results_path + evalutation_time + "_metrics.csv",
            columns=mean_metrics_df.columns,
            index_label=False,
            index=False,
        )
        combined_forecast_df.to_csv(
            results_path + evalutation_time + "_forecast.csv",
            columns=combined_forecast_df.columns,
            index_label=False,
            index=False,
        )

    return mean_metrics_df, combined_forecast_df, model


def multi_train(cfg, experiment, save_model, save_results=True, save_individual=True):
    """
    Deals with instances where multiple models require training at once to be combined.
    :param cfg: project config
    """
    # number of model trained
    if "CEEMDAN" in cfg["TRAIN"]["DECOMPOSITION"]:
        no_trains = cfg["DATA"]["CEEMDAN_DECOMPOSITIONS"]
        # alteration to change dataset used to residuals of dwt
        cfg["TRAIN"]["DECOMPOSITION"].append("CEEMDAN_0")
    elif "DWT" in cfg["TRAIN"]["DECOMPOSITION"]:
        no_trains = 2
        # alteration to change dataset used to residuals of dwt
        cfg["TRAIN"]["DECOMPOSITION"].append("DWT_APPROX")

    for i in range(no_trains):
        # label of data for train
        data_label = cfg["TRAIN"]["DECOMPOSITION"][-1]

        # Conduct the desired train experiment
        if experiment == "train_single":
            _, forecast_df, model = train_single(
                cfg, save_model=save_model, save_metrics=True, fixed_test_set=True
            )
        elif experiment == "train_all":
            _, forecast_df = train_all(cfg, save_models=save_model)
        elif experiment == "hparam_search":
            bayesian_hparam_optimization(cfg)
        elif experiment == "cross_validation":
            _, forecast_df, model = cross_validation(
                cfg,
                save_results=save_results,
                fixed_test_set=True,
                save_individual=save_individual,
            )
        elif experiment == "multi_cross_validation":
            _, forecast_df, model = multi_cross_validation(
                cfg,
                save_results=save_results,
                fixed_test_set=True,
                save_individual=True,
            )
        else:
            raise Exception("Invalid entry in TRAIN > EXPERIMENT field of config.yml.")

        # update series to fit model for
        if "CEEMDAN" in cfg["TRAIN"]["DECOMPOSITION"]:
            # alteration to change dataset used to residuals of dwt
            cfg["TRAIN"]["DECOMPOSITION"].append(f"CEEMDAN_{int(data_label[-1])+1}")
        elif "DWT" in cfg["TRAIN"]["DECOMPOSITION"]:
            # alteration to change dataset used to residuals of dwt
            cfg["TRAIN"]["DECOMPOSITION"].append("DWT_RESIDUAL")

        # concatenate results
        if experiment != "hparam_search":
            if i == 0:
                total_forecast_df = forecast_df
            else:
                total_forecast_df = total_forecast_df + forecast_df

    if save_results and (experiment != "hparam_search"):
        # get results and save
        save_dir = f"{cfg['PATHS']['EXPERIMENTS']}/{cfg['DATA']['SAVE_LABEL']}/"
        model.name += "_combined"
        test_metrics, forecast_df = model.evaluate_forecast(
            total_forecast_df, save_dir=save_dir, plot=True
        )
        logging.info("Test forecast metrics: %s", test_metrics)

    return


def train_experiment(
    cfg=None, experiment="single_train", save_model=False, write_logs=False
):
    """
    Run a training experiment
    :param cfg: Project config
    :param experiment: String defining which experiment to run
    :param save_model: Flag indicating whether to save any models trained during the experiment
    :param write_logs: Flag indicating whether to write logs for training
    """

    # Load project config data
    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", "r"))

    try:
        os.mkdir(f"{cfg['PATHS']['EXPERIMENTS']}/{cfg['DATA']['SAVE_LABEL']}")
    except:
        None

    # Conduct the desired train experiment
    if (cfg["TRAIN"]["DECOMPOSITION"] == ["CEEMDAN"]) or (
        cfg["TRAIN"]["DECOMPOSITION"] == ["DWT"]
    ):
        multi_train(cfg, experiment, save_model)
    elif experiment == "train_single":
        train_single(cfg, save_model=save_model, save_metrics=True, fixed_test_set=True)
    elif experiment == "train_all":
        train_all(cfg, save_models=save_model)
    elif experiment == "hparam_search":
        bayesian_hparam_optimization(cfg)
    elif experiment == "cross_validation":
        cross_validation(
            cfg, save_results=True, fixed_test_set=True, save_individual=True
        )
    elif experiment == "multi_cross_validation":
        multi_cross_validation(cfg, save_results=True, fixed_test_set=True)

    else:
        raise Exception("Invalid entry in TRAIN > EXPERIMENT field of config.yml.")

    # save config to experiment folder
    with open(
        f"{cfg['PATHS']['EXPERIMENTS']}/{cfg['DATA']['SAVE_LABEL']}/{cfg['DATA']['CURRENT_DMA']}_{experiment}_config.yml",
        "w",
    ) as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    return


if __name__ == "__main__":
    cfg = yaml.full_load(open("./config.yml", "r"))
    start = time.time()
    # run for each dma
    if cfg["DATA"]["DMAS"] != None:
        for dma in cfg["DATA"]["DMAS"]:
            logging.info("Running for DMA %s", dma)
            cfg["DATA"]["CURRENT_DMA"] = dma
            train_experiment(
                cfg=cfg,
                experiment=cfg["TRAIN"]["EXPERIMENT"],
                save_model=True,
                write_logs=True,
            )
    else:
        train_experiment(
            cfg=cfg,
            experiment=cfg["TRAIN"]["EXPERIMENT"],
            save_model=True,
            write_logs=True,
        )
    end = time.time() - start
    logging.info("Total time: %f seconds", end)
