import pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import pandas as pd
import os
from models.model import ModelStrategy

class SARIMAXModel(ModelStrategy):
    '''
    A class for a Seasonal Autoregressive Integrated Moving Average Model and the standard operations on it
    '''

    def __init__(self, hparams, log_dir=None):
        univariate = hparams.get('UNIVARIATE', False)
        model = None
        name = 'SARIMAX'
        self.auto_params = hparams.get('AUTO_PARAMS', False)
        self.trend_p = int(hparams.get('TREND_P', 10))
        self.trend_d = int(hparams.get('TREND_D', 2))
        self.trend_q = int(hparams.get('TREND_Q', 0))
        self.seasonal_p = int(hparams.get('SEASONAL_P', 5))
        self.seasonal_d = int(hparams.get('SEASONAL_D', 2))
        self.seasonal_q = int(hparams.get('SEASONAL_Q', 0))
        self.preprocesses = hparams.get('PREPROCESS', ['diurnal'])
        self.m = int(hparams.get('M', 24))
        super(SARIMAXModel, self).__init__(model, univariate, name, log_dir=log_dir)


    def fit(self, dataset):
        '''
        Fits a SARIMAX forecasting model
        :param dataset: A Pandas DataFrame with 2 columns: Date and Consumption
        '''
        if ('Diurnal' in dataset.columns) and ('diurnal' in self.preprocesses):
            print('Using diurnal residual for SARIMAX model')
            dataset = dataset.rename(columns={'Date': 'ds', 'Consumption': 'y', 'Diurnal': 'd'})
            exog = dataset[[col for col in dataset.columns if col not in ['y', 'd']]]
            exog_series = exog.set_index('ds').copy()
            series = dataset.set_index('ds').copy()
            series = series['y'] - series['d']
        else: 
            dataset = dataset.rename(columns={'Date': 'ds', 'Consumption': 'y'})
            if dataset.shape[1] != 2 and self.univariate:
                print('Univariate models cannot fit with datasets with more than 1 feature.')
                dataset = dataset[['ds', 'y']]                  
            else: 
                exog = dataset[[col for col in dataset.columns if col not in ['y', 'd']]]
                exog_series = exog.set_index('ds').copy()
                dataset = dataset[['ds', 'y']] 

            series = dataset.set_index('ds').copy()

        print(series)
        if self.auto_params:
            if self.univariate: 
                best_model = pmdarima.auto_arima(series, seasonal=True, stationary=False, m=self.m, information_criterion='aic',
                                                max_order=2*(self.trend_p + self.trend_q), max_p=2*self.trend_p, max_d=2*self.trend_d,
                                                max_q=2*self.trend_q, max_P=2*self.seasonal_p, max_D=2*self.seasonal_d, max_Q=2*self.seasonal_q,
                                                error_action='ignore')     # Automatically determine model parameters
            else: 
                best_model = pmdarima.auto_arima(series, seasonal=True, stationary=False, m=self.m, information_criterion='aic',
                                                max_order=2*(self.trend_p + self.trend_q), max_p=2*self.trend_p, max_d=2*self.trend_d,
                                                max_q=2*self.trend_q, max_P=2*self.seasonal_p, max_D=2*self.seasonal_d, max_Q=2*self.seasonal_q,
                                                error_action='ignore', exog=exog_series)     # Automatically determine model parameters
            order = best_model.order
            seasonal_order = best_model.seasonal_order
            print("Best SARIMAX params: (p, d, q):", best_model.order, " and  (P, D, Q, s):", best_model.seasonal_order)
        else:
            order = (self.trend_p, self.trend_d, self.trend_q)
            seasonal_order = (self.seasonal_p, self.seasonal_d, self.seasonal_q, self.m)
        
        if self.univariate: 
            self.model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                    enforce_stationarity=True, enforce_invertibility=True).fit()
        else: 
            self.model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                    enforce_stationarity=True, enforce_invertibility=True, exog=exog_series).fit()
        print(self.model.summary())
        return


    def evaluate(self, train_set, test_set, save_dir=None, plot=False):
        '''
        Evaluates performance of SARIMAX model on test set
        :param train_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param test_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param save_dir: Directory in which to save forecast metrics
        :param plot: Flag indicating whether to plot the forecast evaluation
        '''
        if 'diurnal' in self.preprocesses:
            train_set = train_set.rename(columns={'Date': 'ds', 'Consumption': 'y', 'Diurnal': 'd'})
            test_set = test_set.rename(columns={'Date': 'ds', 'Consumption': 'y', 'Diurnal': 'd'})
            train_set = train_set.set_index('ds')
            test_set = test_set.set_index('ds')
            train_exog = train_set[[col for col in train_set.columns if col not in ['y', 'd']]]
            test_exog = test_set[[col for col in test_set.columns if col not in ['y', 'd']]]
            train_set["model"] = self.model.fittedvalues + train_set['d'].values

            if self.univariate: 
                test_set["forecast"] = (self.forecast(test_set.shape[0])['Consumption'].values + test_set['d'].values).tolist()
            else: 
                test_set["forecast"] = (self.forecast(test_set.shape[0], exog=test_exog, exog_forecast=True)['Consumption'].values + test_set['d'].values).tolist()

            df_forecast = pd.concat([train_set, test_set]).rename(columns={'y': 'gt'})
            df_forecast['test_pred'] = 0
            
            test_metrics, df_forecast = self.evaluate_forecast(df_forecast, save_dir=save_dir, plot=plot)
        else: 
            train_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
            test_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
            train_set = train_set.set_index('ds')
            test_set = test_set.set_index('ds')
            train_exog = train_set[[col for col in train_set.columns if col not in ['y', 'd']]]
            test_exog = test_set[[col for col in test_set.columns if col not in ['y', 'd']]]
            train_set["model"] = self.model.fittedvalues

            if self.univariate: 
                test_set["forecast"] = self.forecast(test_set.shape[0])['Consumption'].tolist()
            else: 
                test_set["forecast"] = self.forecast(test_set.shape[0], exog=test_exog, exog_forecast=True)['Consumption'].tolist()

            df_forecast = pd.concat([train_set, test_set]).rename(columns={'y': 'gt'})
            df_forecast['test_pred'] = 0
            test_metrics, df_forecast = self.evaluate_forecast(df_forecast, save_dir=save_dir, plot=plot)

        return test_metrics, df_forecast


    def forecast(self, days, exog=None, exog_forecast=False, recent_data=None):
        '''
        Create a forecast for the test set. Note that this is different than obtaining predictions for the test set.
        The model makes a prediction for the provided example, then uses the result for the next prediction.
        Repeat this process for a specified number of days.
        :param days: Number of days into the future to produce a forecast for
        :param recent_data: A factual example for the first prediction
        :return: An array of predictions
        '''
        if exog_forecast: 
            forecast_df = self.model.forecast(steps=days, exog=exog).reset_index(level=0)
        else:
            forecast_df = self.model.forecast(steps=days).reset_index(level=0)
        forecast_df.columns = ['Date', 'Consumption']
        return forecast_df


    def save(self, save_dir, scaler_dir=None):
        '''
        Saves the model to disk
        :param save_dir: Directory in which to save the model
        '''
        if self.model:
            model_path = os.path.join(save_dir, self.name + self.train_date + '.pkl')
            self.model.save(model_path)  # Serialize and save the model object


    def load(self, model_path, scaler_path=None):
        '''
        Loads the model from disk
        :param model_path: Path to saved model
        '''
        if os.path.splitext(model_path)[1] != '.pkl':
            raise Exception('Model file path for ' + self.name + ' must have ".pkl" extension.')
        self.model = SARIMAXResults.load(model_path)
        return