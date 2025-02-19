import pandas as pd
import yaml
import glob
import numpy as np
import pywt
from datetime import datetime
from tqdm import tqdm
from statsmodels.tsa.seasonal import MSTL
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from PyEMD import CEEMDAN
import logging
import os


# Load project configuration
cfg = yaml.full_load(open("./config.yml", 'r'))       # Load project config data
# set up logging
# create log directory if it doesn't exist
log_dir = os.path.abspath(cfg['PATHS']['LOG_DIR']+'data_prep/.')
os.makedirs(log_dir, exist_ok=True)

# configure logging
log_filename = os.path.join(log_dir, f'log_{datetime.now().strftime("%Y-%m-%d")}.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_filename),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

def load_raw_data(cfg, save_raw_df=True, rate_class='all'):
    """Load all entries for water consumption and combine into a single dataframe.

    Args:
        cfg (dict): Project configuration.
        save_raw_df (bool): Flag indicating whether to save the accumulated raw dataset.
        rate_class (str): Rate class to filter raw data by.

    Returns:
        tuple: A tuple containing two Pandas dataframes, one for water consumption records and one for exogenous variables.
    """

    # demand data
    raw_data_filenames = glob.glob(cfg['PATHS']['RAW_DATA_DIR'] + "/*.csv")
    logger.info('Loading raw data from spreadsheets.')
    raw_df = pd.DataFrame()
    for filename in tqdm(raw_data_filenames):
        df = pd.read_csv(filename, low_memory=False, index_col=False)    # Load a water demand CSV
        df = df.set_index('Date')
        df.index.name = 'Date'
        # allow python to interpret dates
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        raw_df = pd.concat([raw_df, df], axis=0, ignore_index=False)     # Concatenate next batch of data
        shape1 = raw_df.shape
        raw_df = raw_df[~raw_df.index.duplicated(keep='first')]   # Drop duplicate entries appearing in different data slices
        print("Deduplication: ", shape1, "-->", raw_df.shape)
    
    logger.info('Consumption total: %d', len(raw_df))
    logger.info('Original data shape: %s', raw_df.shape)

    # exogenous variable data
    raw_exog_filenames = glob.glob(cfg['PATHS']['RAW_EXOG_DIR'] + "/*.csv")
    logger.info('Loading exogenous data from spreadsheets.')
    exog_df = pd.DataFrame()
    for filename in tqdm(raw_exog_filenames):
        df = pd.read_csv(filename, low_memory=False, index_col=False)    # Load a water demand CSV
        df = df.set_index('Date')
        df.index.name = 'Date'
        # correct this if date format is wrong
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S") 
        exog_df = pd.concat([exog_df, df], axis=0, ignore_index=False)     # Concatenate next batch of data
        shape1 = raw_df.shape
        exog_df = exog_df[~exog_df.index.duplicated(keep='first')]   # Drop duplicate entries appearing in different data slices

    if save_raw_df:
        raw_df.to_csv(cfg['PATHS']['RAW_DATASET'], header=True, index_label='Date', index=True)
        exog_df.to_csv(cfg['PATHS']['RAW_EXOG_DATASET'], header=True, index_label='Date', index=True)
    return raw_df, exog_df


def impute_ts(dataset, corr_thresh=0.7, method='hybrid'):
    """Imputes data using kNN algorithm for correlated DMAs.

    Args:
        dataset (pd.DataFrame): Dataframe to impute.
        corr_thresh (float): Threshold for correlation to be considered a partner DMA.
        method (str): Imputation method to use ('hybrid' or 'mean').

    Returns:
        pd.DataFrame: Imputed dataframe.
    """
    logger.info('Applying imputation...')

    if method == 'hybrid': 
        # Find comparison partners
        correlation = dataset.corr()
        correlation = correlation[correlation!=1]
        partners = {
            dma: list(correlation[(correlation[dma]>=corr_thresh)].index) 
            for dma in correlation.columns
        }

        # scale
        scaler = {}
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(      
                n_estimators=10,
                max_depth=10,
                bootstrap=True,
                max_samples=0.5,
                n_jobs=2,
                random_state=0
            ),
            max_iter=20, 
            tol=0.005
        )

        # pre allocate
        scaled_df = pd.DataFrame(columns=dataset.columns, index=dataset.index)
        imputed_df = pd.DataFrame(columns=dataset.columns, index=dataset.index)

        # apply scaling
        for dma in dataset.columns: 
            values = dataset[dma].values
            values = values.reshape((len(values), 1))
            # train the standardization
            scaler[dma] = StandardScaler()
            scaler[dma] = scaler[dma].fit(values)
            standardized = scaler[dma].transform(values)
            scaled_df[dma] = standardized.flatten()
            
        # impute
        for dma in dataset.columns:
            logger.info(f"Applying imputation for {dma}...")
            # apply missForest imputation
            cols = partners[dma]+[dma]  # correlated partner set
            fit = imputer.fit_transform(scaled_df[cols])
            imputed_df[dma] = pd.DataFrame(fit, columns=cols, index=dataset.index)[dma]

            # inverse scaling
            values = imputed_df[dma].values
            values = values.reshape((len(values), 1))
            standardized = scaler[dma].inverse_transform(values)
            imputed_df[dma] = standardized.flatten()

            # apply seasonal decomp
            stl = MSTL(imputed_df[dma].values.reshape(-1), periods=(24, 24*7))
            res = stl.fit()

            # Extract the seasonal and trend components
            seasonal_component = res.seasonal.sum(axis=1) + res.trend
            # Create the deseasonalised series with original dataset
            df_deseasonalised = dataset[dma] - seasonal_component

            # Interpolate missing values in the deseasonalised series
            df_deseasonalised_imputed = df_deseasonalised.interpolate(method="linear", limit=50)
            df_deseasonalised_imputed = df_deseasonalised.fillna(0)

            # Add the seasonal component back to create the final imputed series
            imputed_df[dma] = df_deseasonalised_imputed + seasonal_component
        
    elif method == 'mean': 
        imputed_df = dataset.copy()
        # exponential mean component
        for dma in dataset.columns:
            avg_df = dataset[[dma]].copy()
            missing_dma_indices = avg_df[avg_df.isna().all(axis=1)].index

            avg_df['mean'] = 0
            
            # fill with nan mean for insufficient data for exponential
            avg_df['day'] = avg_df.index.weekday
            avg_df['hour'] = avg_df.index.hour
            
            avg_values = avg_df.groupby(by=['day', 'hour']).mean()[dma]
                
            for idx, row in avg_df.iterrows(): 
                avg_df.loc[idx, 'mean'] = avg_values.loc[row.day, row.hour]
                
            # residual correction component
            avg_df['residual'] = avg_df[dma] - avg_df['mean']
            avg_df['res_correction'] = avg_df['residual'].rolling(window=48).mean()
            avg_df['mean'] = avg_df['mean'] + avg_df['res_correction'].fillna(how='ffill').fillna(how='bfill')
            
            imputed_df.loc[missing_dma_indices, dma] = avg_df.loc[missing_dma_indices, 'mean']

    return imputed_df

def remove_anomoly(dataset, contamination=0.005):
    """Transform raw water demand data into a time series dataset with anomalous points removed.

    Args:
        dataset (pd.DataFrame): Dataframe with raw data for each DMA with no gaps and datetime index.
        contamination (float): Percentage level of anomalous data points in training dataset.

    Returns:
        pd.DataFrame: Dataframe with anomalies removed and imputed.
    """
    logger.info('Removing anomalies...')
    # pre allocate
    df_anomoly = dataset.copy()

    # split data into each dma
    for dma in dataset.columns: 
        df_anomoly['hour'] = df_anomoly.index.hour
        # split dma data into hourly values
        for hour in range(24): 
            # group by hour
            vals = df_anomoly[df_anomoly['hour']==hour][[dma]]
            clf = IsolationForest(contamination=contamination, random_state=0).fit(vals.values.reshape(-1, 1))
            vals['outlier'] = clf.predict(vals.values.reshape(-1, 1))
            # get index of outliers
            outlier_idx = vals[vals['outlier']==-1].index
            # set outliers to np.nan for imputation
            df_anomoly.loc[outlier_idx, dma] = np.nan

    df_anomoly = df_anomoly.drop(columns=['hour'])

    logger.info('Imputing anomalous points...')
    df_anomoly = impute_ts(df_anomoly)

    return df_anomoly


def dwt_transform(dataset, dmas, levels=1):
    """Applies discrete wavelet transformation to extract approximation series.

    Args:
        dataset (pd.DataFrame): Dataframe with raw data for each DMA with no gaps and datetime index.
        dmas (list): List of DMAs to transform.
        levels (int): Number of levels of decomposition for approximation coefficients.

    Returns:
        pd.DataFrame: Dataframe with approximate series for each DMA.
    """
    logger.info('Applying DWT transform...')
    # scale data
    standard_scaler = StandardScaler()
    df = dataset.copy()
    df.loc[:, dmas] = standard_scaler.fit_transform(df.loc[:, dmas])

    # apply dwt to each column
    for dma in dmas: 
        logger.info(f"Applying DWT transform for {dma}...")
        coeffs = pywt.wavedec(df[dma].values, 'db4', mode='symmetric', level=levels)
        approx = pywt.idwt(cA=coeffs[0], cD=None ,wavelet='db4', mode='symmetric')
        df[dma] = approx
    
    # revert scaling
    df.loc[:, dmas] = standard_scaler.inverse_transform(df.loc[:, dmas])

    return df

 
def ceemdan_transform(dataset, dmas, sets=3):
    """Applies CEEMDAN transformation to extract approximation series.

    Args:
        dataset (pd.DataFrame): Dataframe with raw data for each DMA with no gaps and datetime index.
        dmas (list): List of DMAs to transform.
        sets (int): Number of CEEMDAN components to extract.

    Returns:
        dict: Dictionary containing dataframes of each transformed series.
    """
    logger.info('Applying CEEMDAN transform...')
    # scale data
    standard_scaler = StandardScaler()

    # dict containing df of each transformed series
    transform_sets = {i: dataset.copy() for i in range(sets+1)}

    # apply ceemdan to each column
    for dma in dmas: 
        logger.info(f"Applying CEEMDAN transform for {dma}...")
        ceemdan = CEEMDAN(epsilon=0.2)
        transformed = ceemdan(dataset[dma].values, max_imf=sets).T

        # save each transform 
        for set in range(sets+1): 
            transform_sets[set][dma] = transformed[:, set]

    return transform_sets


def diurnal_flow(cfg, dataset, stat='mean'):
    """Add diurnal flow as mean value of each day and hour to original dataset.

    Args:
        cfg (dict): Project configuration.
        dataset (pd.DataFrame): Dataframe with raw data for each DMA with no gaps and datetime index.
        stat (str): Statistic to use for diurnal flow ('mean' or 'median').

    Returns:
        pd.DataFrame: Dataframe with diurnal flow added.
    """
    logger.info('Adding diurnal flow...')
    # exponential mean component
    avg_df = pd.DataFrame(columns=dataset.columns, index=dataset.index)
    dmas = dataset.columns
    df = dataset.copy()
    
    # fill with nan mean for insufficient data for exponential
    df['day'] = df.index.weekday
    df['hour'] = df.index.hour
    
    for dma in dmas: 
        if stat == 'mean': 
            avg_values = df[:-cfg['DATA']['TEST_DAYS']].groupby(by=['day', 'hour']).mean()[dma]
        elif stat == 'median': 
            avg_values = df[:-cfg['DATA']['TEST_DAYS']].groupby(by=['day', 'hour']).median()[dma]

        for idx, row in df.iterrows(): 
            avg_df.loc[idx, dma] = avg_values.loc[row.day, row.hour]

    avg_df = avg_df.rename(columns={dma : 'diurnal_'+dma for dma in dmas})

    # shift forward to give next diurnal value
    #avg_df = avg_df.shift(-1).fillna(0)

    return avg_df


def load_exog(cfg, dataset, exog):
    """Add time features to dataframe.

    Args:
        cfg (dict): Project configuration.
        dataset (pd.DataFrame): Dataframe with raw data for each DMA with no gaps and datetime index.
        exog (pd.DataFrame): Dataframe with exogenous variables.

    Returns:
        pd.DataFrame: Dataframe with time features added.
    """
    logger.info('Loading exogenous data...')
    time_feats = cfg['DATA']['TIME_FEATS']
    weather_feats = cfg['DATA']['WEATHER_FEATS']

    # fill with nan mean for insufficient data for exponential
    if 'WEEKDAY' in time_feats: 
        dataset['WEEKDAY'] = dataset.index.weekday
        dataset['WEEKDAY'] = dataset['WEEKDAY'] / 6

    if 'HOUR' in time_feats: 
        dataset['HOUR'] = dataset.index.hour
        dataset['HOUR'] = dataset['HOUR'] / 23

    if 'HOLIDAY' in time_feats: 
        # make list of special days for the DMAs region
        holidays = cfg['DATES']['HOLIDAYS']['Official Holiday']
        holidays += cfg['DATES']['HOLIDAYS']['Event Day']
        #holidays += cfg['DATES']['HOLIDAYS'].get('Unofficial Holiday', [])
        # make columns for special days
        dataset['HOLIDAY'] = 0

        # add indicator variable for special days
        for i in dataset.index:
            if str(i)[:10] in holidays:
                dataset.loc[i, 'HOLIDAY'] = 1

    if 'WEEKEND' in time_feats: 
        # add variable for weekend days
        dataset['WEEKEND'] = 0

        for i in dataset.index:
            if i.weekday() == 5 or i.weekday() == 6:
                dataset.loc[i, 'WEEKEND'] = 1

    for feat in weather_feats: 
        dataset[feat] = exog[feat]

    return dataset


def preprocess_ts(cfg=None, save_raw_df=True, save_prepr_df=True, rate_class='all', out_path=None, save_split_prepr_df=True):
    """Transform raw water demand data into a time series dataset ready to be fed into a model.

    Args:
        cfg (dict, optional): Project configuration. Defaults to None.
        save_raw_df (bool): Flag indicating whether to save intermediate raw data.
        save_prepr_df (bool): Flag indicating whether to save the preprocessed data.
        rate_class (str): Rate class to filter by.
        out_path (str, optional): Path to save updated preprocessed data. Defaults to None.
        save_split_prepr_df (bool): Flag indicating whether to save split preprocessed data.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    run_start = datetime.today()
    tqdm.pandas()

    weather_feats = cfg['DATA']['WEATHER_FEATS']
    time_feats = cfg['DATA']['TIME_FEATS']
    specific_feats = cfg['DATA']['SPECIFIC_FEATS']
    feat_names = weather_feats + time_feats + specific_feats
    
    preprocesses = cfg['DATA']['PREPROCESS']

    # remove unwanted data
    raw_df, exog_df = load_raw_data(cfg, rate_class=rate_class, save_raw_df=save_raw_df)
    dmas = cfg['DATA']['DMAS']  # list of DMAs to preprocess

    if cfg['DATA']['END_TRIM'] > 0: 
        preprocessed_df = raw_df[cfg['DATA']['START_TRIM']:-cfg['DATA']['END_TRIM']]
    else: 
        preprocessed_df = raw_df[cfg['DATA']['START_TRIM']:]
    logger.info('Trimmed data shape: %s', preprocessed_df.shape)

    # impute dataset with seasonal decomposition and missForest
    if 'IMPUTE' in preprocesses: 
        preprocessed_df = impute_ts(preprocessed_df)
        exog_df = impute_ts(exog_df)
    
    # remove anomolies with isolation forest
    if 'ANOMOLY_REMOVAL' in preprocesses: 
        preprocessed_df = remove_anomoly(preprocessed_df)

    if 'DIURNAL' in specific_feats: 
        diurnal_df = diurnal_flow(cfg, preprocessed_df)
        preprocessed_df = pd.concat([preprocessed_df, diurnal_df], axis=1)

    if (time_feats != None) or (weather_feats != None): 
        print(exog_df)
        preprocessed_df = load_exog(cfg, preprocessed_df, exog_df)

    if 'DWT' in preprocesses: 
        dwt_df = dwt_transform(preprocessed_df, dmas, levels=cfg['DATA']['DWT_DECOMPOSITIONS'])
        residuals_df = preprocessed_df - dwt_df
    
    if 'CEEMDAN' in preprocesses: 
        preprocessed_transforms = ceemdan_transform(preprocessed_df, dmas, sets=cfg['DATA']['CEEMDAN_DECOMPOSITIONS']-1)

    if save_prepr_df:
        save_path = cfg['PATHS']['PREPROCESSED_DATA'] if out_path is None else out_path
        preprocessed_df.to_csv(save_path, sep=',', header=True)

        if 'DWT' in preprocesses: 
            save_path_dwt = cfg['PATHS']['PREPROCESSED_DWT_DATA'][:-4]+'_dwt_approx.csv' if out_path is None else out_path
            save_path_residuals = cfg['PATHS']['PREPROCESSED_DWT_DATA'][:-4]+'_dwt_residual.csv' if out_path is None else out_path
            
            dwt_df.to_csv(save_path_dwt, sep=',', header=True)
            residuals_df.to_csv(save_path_residuals, sep=',', header=True)
        
        if 'CEEMDAN' in preprocesses: 
            for set in preprocessed_transforms.keys(): 
                save_path_ceemdan = f"{cfg['PATHS']['PREPROCESSED_CEEMDAN_DATA'][:-4]}_ceemdan_{set}.csv" if out_path is None else out_path
                preprocessed_transforms[set].to_csv(save_path_ceemdan, sep=',', header=True)
  
    logger.info('Done. Runtime = %f min', ((datetime.today() - run_start).seconds / 60))
    return preprocessed_df


if __name__ == '__main__':
    logger.info('Running preprocess_ts from directory: %s', os.getcwd())
    df = preprocess_ts(cfg=cfg, rate_class='ins', save_raw_df=True, save_prepr_df=True, save_split_prepr_df=True)
