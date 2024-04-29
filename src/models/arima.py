import pmdarima
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import os
from models.model import ModelStrategy
import pandas as pd

class ARIMAModel(ModelStrategy):
    '''
    A class for an Autoregressive Integrated Moving Average Model and the standard operations on it
    '''

    def __init__(self, hparams, log_dir=None):
        model = None
        name = 'ARIMA'
        self.univariate = hparams.get('UNIVARIATE', True)
        self.auto_params = hparams['AUTO_PARAMS']
        self.p = int(hparams.get('P', 2))
        self.d = int(hparams.get('D', 2))
        self.q = int(hparams.get('Q', 2))
        super(ARIMAModel, self).__init__(model, self.univariate, name, log_dir=log_dir)


    def fit(self, dataset):
        '''
        Fits an ARIMA forecasting model
        :param dataset: A Pandas DataFrame with 2 columns: Date and Consumption
        '''
        if dataset.shape[1] > 3:
            raise Exception('Univariate models cannot fit with datasets with more than 2 features.')
        if self.univariate: 
            dataset.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
            series = dataset.set_index('ds')
        else: 
            dataset.rename(columns={'Date': 'ds', 'Consumption': 'y', 'Exog': 'x'}, inplace=True)
            series = dataset.set_index('ds')[['y']]
            x = dataset.set_index('ds')[['x']]

        if self.auto_params:
            if self.univariate:
                best_model = pmdarima.auto_arima(series, seasonal=False, stationary=False, information_criterion='aic',
                                                max_order=2*(self.p + self.q), max_p=2*self.p, max_d=2*self.d,
                                                max_q=2*self.q, error_action='ignore')
            else: 
                best_model = pmdarima.auto_arima(series, seasonal=False, stationary=False, information_criterion='aic',
                                                max_order=2*(self.p + self.q), max_p=2*self.p, max_d=2*self.d,
                                                max_q=2*self.q, error_action='ignore', X=x)
            order = best_model.order
            print("Best ARIMA params: (p, d, q):", best_model.order)
        else:
            order = (self.p, self.d, self.q)
        
        if self.univariate: 
            self.model = ARIMA(series, order=order).fit()
        else: 
            self.model = ARIMA(series, order=order, exog=x).fit()
        print(self.model.summary())
        return


    def evaluate(self, train_set, test_set, save_dir=None, plot=False):
        '''
        Evaluates performance of ARIMA model on test set
        :param train_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param test_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param save_dir: Directory in which to save forecast metrics
        :param plot: Flag indicating whether to plot the forecast evaluation
        '''
        if self.univariate: 
            train_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
            test_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
            train_set = train_set.set_index('ds')
            test_set = test_set.set_index('ds')
            train_set["model"] = self.model.fittedvalues
            test_set["forecast"] = self.forecast(test_set.shape[0])['Consumption'].tolist()
        else: 
            train_set.rename(columns={'Date': 'ds', 'Consumption': 'y', 'Exog': 'x'}, inplace=True)
            test_set.rename(columns={'Date': 'ds', 'Consumption': 'y', 'Exog': 'x'}, inplace=True)
            train_set = train_set.set_index('ds')
            test_set = test_set.set_index('ds')
            train_set["model"] = self.model.fittedvalues
            test_set["forecast"] = self.forecast(test_set['y'].shape[0], x=test_set['x'])['Consumption'].tolist()

        df_forecast = pd.concat([train_set, test_set]).rename(columns={'y': 'gt'})
        test_metrics = self.evaluate_forecast(df_forecast, save_dir=save_dir, plot=plot)
        return test_metrics


    def forecast(self, days, x, recent_data=None):
        '''
        Create a forecast for the test set. Note that this is different than obtaining predictions for the test set.
        The model makes a prediction for the provided example, then uses the result for the next prediction.
        Repeat this process for a specified number of days.
        :param days: Number of days into the future to produce a forecast for
        :param recent_data: A factual example for the first prediction
        :param x: exogenous variable
        :return: An array of predictions
        '''
        if self.univariate: 
            forecast_df = self.model.forecast(steps=days).reset_index(level=0)
        else: 
            forecast_df = self.model.forecast(steps=days, exog=x).reset_index(level=0)
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
        self.model = ARIMAResults.load(model_path)
        return