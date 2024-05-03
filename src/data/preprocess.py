import os
import pandas as pd
import yaml
import glob
import numpy as np
import pywt
from datetime import datetime, timedelta
from tqdm import tqdm
from statsmodels.tsa.seasonal import MSTL
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from PyEMD import CEEMDAN

def load_raw_data(cfg, save_raw_df=True, rate_class='all'):
    '''
    Load all entries for water consumption and combine into a single dataframe
    :param cfg: project config
    :param save_raw_df: Flag indicating whether to save the accumulated raw dataset
    :param rate_class: Rate class to filter raw data by
    :return: a Pandas dataframe containing all water consumption records
    '''

    weather_feats = cfg['DATA']['WEATHER_FEATS']
    frequency = cfg['DATA']['FREQUENCY']

    raw_data_filenames = glob.glob(cfg['PATHS']['RAW_DATA_DIR'] + "/*.csv")
    print('Loading raw data from spreadsheets.')
    raw_df = pd.DataFrame()
    for filename in tqdm(raw_data_filenames):
        df = pd.read_csv(filename, low_memory=False, index_col=False)    # Load a water demand CSV
        df = df.set_index('Date')
        df.index.name = 'Date'
        df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M")
        raw_df = pd.concat([raw_df, df], axis=0, ignore_index=False)     # Concatenate next batch of data
        shape1 = raw_df.shape
        raw_df = raw_df[~raw_df.index.duplicated(keep='first')]   # Drop duplicate entries appearing in different data slices
        print("Deduplication: ", shape1, "-->", raw_df.shape)
    
    print('Consumption total: ', len(raw_df))
    print(f"Original data shape: {raw_df.shape}")

    if save_raw_df:
        raw_df.to_csv(cfg['PATHS']['RAW_DATASET'], header=True, index_label='Date', index=True)
    return raw_df


def impute_ts(dataset, corr_thresh=0.7):
    """
    Imputes data using kNN alogrithm for correlated dmas
    :param df: dataframe to impute
    :param k: number of neighbours
    :corr_thresh: a float corresponding to threshold for correlation to be considered a partner dma
    """
    print('Applying imputation...')
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
        print(f"Applying imputation for {dma}...")
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
        df_deseasonalised_imputed = df_deseasonalised.interpolate(method="linear")
        df_deseasonalised_imputed = df_deseasonalised.fillna(0)

        # Add the seasonal component back to create the final imputed series
        imputed_df[dma] = df_deseasonalised_imputed + seasonal_component

    return imputed_df

def remove_anomoly(dataset, contamination=0.005):
    '''
    Transform raw water demand data into a time series dataset with anomolous points removed.
    :param cfg: project config
    :param dataset: Dataframe with raw data for each dma with no gaps and datetime index
    :param contamination: % level of anomolous data points in training dataset
    '''
    print('Removing anomolies...')
    # pre allocate
    df_anomoly = dataset.copy()

    # split data into each dma
    for dma in dataset.columns: 
        df_anomoly['hour'] = df_anomoly.index.hour
        # split dma data into hourly values
        for hour in range(24): 
            # group by hour
            vals = df_anomoly[df_anomoly['hour']==hour][[dma]]
            clf = IsolationForest(random_state=0, contamination=0.005).fit(vals.values.reshape(-1, 1))
            vals['outlier'] = clf.predict(vals.values.reshape(-1, 1))
            # get index of outliers
            outlier_idx = vals[vals['outlier']==-1].index
            # set outliers to np.nan for imputation
            df_anomoly.loc[outlier_idx, dma] = np.nan

    df_anomoly = df_anomoly.drop(columns=['hour'])

    print('Imputing anomolous points...')
    df_anomoly = impute_ts(df_anomoly)

    return df_anomoly


def dwt_transform(dataset, dmas, levels=1):
    '''
    Applies discrete wavelet transformation to extract approximation series
    :param dataset: Dataframe with raw data for each dma with no gaps and datetime index
    :levels: Number of levels of decomposition for approx coefficients
    :returns: Dataframe with approximate series for each dma
    '''
    # scale data
    standard_scaler = StandardScaler()
    df = dataset.copy()
    df.loc[:, dmas] = standard_scaler.fit_transform(df.loc[:, dmas])

    # apply dwt to each column
    for dma in dmas: 
        print(f"Applying DWT transform for {dma}...")
        coeffs = pywt.wavedec(df[dma].values, 'db4', mode='symmetric', level=levels)
        approx = pywt.idwt(cA=coeffs[0], cD=None ,wavelet='db4', mode='symmetric')
        df[dma] = approx
    
    # revert scaling
    df.loc[:, dmas] = standard_scaler.inverse_transform(df.loc[:, dmas])

    return df

 
def ceemdan_transform(dataset, dmas, sets=3):
    '''
    Applies discrete wavelet transformation to extract approximation series
    :param dataset: Dataframe with raw data for each dma with no gaps and datetime index
    :sets: Number of ceemdan components to extract
    :returns: Dataframe with approximate series for each dma
    '''

    # scale data
    standard_scaler = StandardScaler()

    # dict containing df of each transformed series
    transform_sets = {i: dataset.copy() for i in range(sets+1)}

    # apply ceemdan to each column
    for dma in dmas: 
        print(f"Applying CEEMDAN transform for {dma}...")
        ceemdan = CEEMDAN(epsilon=0.2)
        transformed = ceemdan(dataset[dma].values, max_imf=sets).T

        # save each transform 
        for set in range(sets+1): 
            transform_sets[set][dma] = transformed[:, set]

    return transform_sets


def diurnal_flow(cfg, dataset, stat='mean'):
    '''
    Add diurnal flow as mean value of each day and hour to original dataset
    :param cfg: project config
    :param dataset: Dataframe with raw data for each dma with no gaps and datetime index
    '''
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
    avg_df = avg_df.shift(-1).fillna(0)

    return avg_df


def time_features(cfg, dataset):
    '''
    Add time features to dataframe
    :param cfg: project config
    :param dataset: Dataframe with raw data for each dma with no gaps and datetime index
    '''

    time_feats = cfg['DATA']['TIME_FEATS']

    # fill with nan mean for insufficient data for exponential
    if 'WEEKDAY' in time_feats: 
        dataset['WEEKDAY'] = dataset.index.weekday

    if 'HOUR' in time_feats: 
        dataset['HOUR'] = dataset.index.hour

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

    return dataset


def preprocess_ts(cfg=None, save_raw_df=True, save_prepr_df=True, rate_class='all', out_path=None, save_split_prepr_df=True):
    '''
    Transform raw water demand data into a time series dataset ready to be fed into a model.
    :param cfg: project config
    :param save_raw_df: Flag indicating whether to save intermediate raw data
    :param save_prepr_df: Flag indicating whether to save the preprocessed data
    :param rate_class: Rate class to filter by
    :param out_path: Path to save updated preprocessed data
    '''

    run_start = datetime.today()
    tqdm.pandas()
    if cfg is None:
        cfg = yaml.full_load(open("./config.yml", 'r'))       # Load project config data

    weather_feats = cfg['DATA']['WEATHER_FEATS']
    time_feats = cfg['DATA']['TIME_FEATS']
    specific_feats = cfg['DATA']['SPECIFIC_FEATS']
    feat_names = weather_feats + time_feats + specific_feats
    
    preprocesses = cfg['DATA']['PREPROCESS']

    # remove unwanted data
    raw_df = load_raw_data(cfg, rate_class=rate_class, save_raw_df=save_raw_df)
    dmas = raw_df.columns

    if cfg['DATA']['END_TRIM'] > 0: 
        preprocessed_df = raw_df[cfg['DATA']['START_TRIM']:-cfg['DATA']['END_TRIM']]
    else: 
        preprocessed_df = raw_df[cfg['DATA']['START_TRIM']:]
    print(f"Trimmed data shape: {preprocessed_df.shape}")

    # impute dataset with seasonal decomposition and missForest
    if 'IMPUTE' in preprocesses: 
        preprocessed_df = impute_ts(preprocessed_df)
    
    # remove anomolies with isolation forest
    if 'ANOMOLY_REMOVAL' in preprocesses: 
        preprocessed_df = remove_anomoly(preprocessed_df)

    if 'DIURNAL' in specific_feats: 
        diurnal_df = diurnal_flow(cfg, preprocessed_df)
        preprocessed_df = pd.concat([preprocessed_df, diurnal_df], axis=1)

    if time_feats != None: 
        preprocessed_df = time_features(cfg, preprocessed_df)

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
    """ 
    # splits processed data into individual dmas
    if save_split_prepr_df:
        for dma in dmas: 
            # add all relevant cols for each dma
            cols = []
            cols += weather_feats + time_feats
            cols += [i for i in preprocessed_df.columns if dma[-1] in i.split('_')]
 
            save_path = cfg['PATHS']['PREPROCESSED_DATA'][:-4]+'_'+dma+'.csv' if out_path is None else out_path[:-4]+'_'+dma+'.csv'
            preprocessed_df[cols].rename(columns={dma: 'Consumption'}).to_csv(save_path, sep=',', header=True)

            if 'DWT' in preprocesses: 
                save_path_dwt = cfg['PATHS']['PREPROCESSED_DWT_DATA'][:-4]+'_'+dma+'_dwt_approx.csv' if out_path is None else out_path[:-4]+'_'+dma+'.csv'
                save_path_residuals = cfg['PATHS']['PREPROCESSED_DWT_DATA'][:-4]+'_'+dma+'_dwt_residual.csv' if out_path is None else out_path[:-4]+'_'+dma+'.csv'
                
                dwt_df[cols].rename(columns={dma: 'Consumption'}).to_csv(save_path_dwt, sep=',', header=True)
                residuals_df[cols].rename(columns={dma: 'Consumption'}).to_csv(save_path_residuals, sep=',', header=True)

            if 'CEEMDAN' in preprocesses: 
                for set in preprocessed_transforms.keys(): 
                    save_path_ceemdan = f"{cfg['PATHS']['PREPROCESSED_CEEMDAN_DATA'][:-4]}_{dma}_ceemdan_{set}.csv" if out_path is None else out_path
                    preprocessed_transforms[set][cols].rename(columns={dma: 'Consumption'}).to_csv(save_path_ceemdan, sep=',', header=True)
    """
    print("Done. Runtime = ", ((datetime.today() - run_start).seconds / 60), " min")
    return preprocessed_df


if __name__ == '__main__':
    df = preprocess_ts(rate_class='ins', save_raw_df=True, save_prepr_df=True, save_split_prepr_df=True)
