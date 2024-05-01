from abc import ABCMeta, abstractmethod
import tensorflow as tf
import math
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.models import save_model, load_model
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import pandas as pd
import numpy as np
import os
from models.model import ModelStrategy
import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class NNModel(ModelStrategy):
    '''
    A class representing a neural network model defined using TensorFlow and the standard operations on it
    '''
    __metaclass__ = ABCMeta

    def __init__(self, hparams, name, log_dir):
        self.univariate = hparams.get('UNIVARIATE', True)
        self.batch_size = int(hparams.get('BATCH_SIZE', 32))
        self.epochs = int(hparams.get('EPOCHS',120))
        self.patience = int(hparams.get('PATIENCE', 15))
        self.val_frac = hparams.get('VAL_FRAC', 0.15)
        self.T_x = int(hparams.get('T_X', 32))
        self.metrics = [MeanSquaredError(name='mse'), RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae'),
                        MeanAbsolutePercentageError(name='mape')]
        self.standard_scaler = StandardScaler()
        self.forecast_start = datetime.datetime.today()
        model = None
        super(NNModel, self).__init__(model, self.univariate, name, log_dir=log_dir)


    @abstractmethod
    def define_model(self, input_dim):
        '''
        Abstract method for TensorFlow model definition
        '''
        pass


    def fit(self, dataset):
        '''
        Fits an RNN forecasting model
        :param dataset: A Pandas DataFrame with feature columns and a Consumption column
        '''
        df = dataset.copy()
        if self.univariate:
            df = df[['Date', 'Consumption']]
        df.loc[:, dataset.columns != 'Date'] = self.standard_scaler.fit_transform(dataset.loc[:, dataset.columns != 'Date'])

        train_df = df[0:-int(df.shape[0]*self.val_frac)]
        val_df = df[-int(df.shape[0]*self.val_frac):]

        # Make time series datasets
        train_df, val_df, train_dates, X_train, Y_train, val_dates, X_val, Y_val = self.preprocess(train_df, val_df)
        # Define model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.define_model(input_shape)

        # Define model callbacks
        callbacks = [EarlyStopping(monitor='val_loss', verbose=1, patience=self.patience, mode='min', restore_best_weights=True)]
        if self.log_dir is not None:
            callbacks.append(TensorBoard(log_dir=os.path.join(self.log_dir, 'training', self.train_date), histogram_freq=1))

        # Train RNN model
        print(X_train.shape)
        history = self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs,
                                 validation_data=(X_val, Y_val), callbacks=callbacks, verbose=1, shuffle=False)
        return


    def evaluate(self, train_set, test_set, save_dir=None, plot=False):
        '''
        Evaluates performance of RNN model on test set
        :param train_set: A Pandas DataFrame with feature columns and a Consumption column
        :param test_set: A Pandas DataFrame with feature columns and a Consumption column
        :param save_dir: Directory in which to save forecast metrics
        :param plot: Flag indicating whether to plot the forecast evaluation
        '''

        if self.univariate:
            train_set = train_set[['Date', 'Consumption']]
            test_set = test_set[['Date', 'Consumption']]

        train_set.loc[:, train_set.columns != 'Date'] = self.standard_scaler.transform(train_set.loc[:, train_set.columns != 'Date'])
        test_set.loc[:, test_set.columns != 'Date'] = self.standard_scaler.transform(test_set.loc[:, test_set.columns != 'Date'])

        # Create windowed versions of the training and test sets
        consumption_idx = train_set.drop('Date', axis=1).columns.get_loc('Consumption')   # Index of consumption feature
        train_set, test_set, train_dates, X_train, Y_train, test_pred_dates, X_test, Y_test = self.preprocess(train_set, test_set)
        test_forecast_dates = test_set['Date']
        train_dates = train_set[-len(train_dates):]['Date']

        # Make predictions for training set and obtain forecast for test set
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        if 'windowed' in self.preprocesses: 
            forecast_data = X_train[-1]
        elif 'fragments' in self.preprocesses: 
            forecast_data = train_set[len(X_train):]

        test_forecast_df = self.forecast(test_forecast_dates.shape[0], recent_data=forecast_data, test_set=test_set)

        # Rescale data
        train_set.loc[:, train_set.columns != 'Date'] = self.standard_scaler.inverse_transform(train_set.loc[:, train_set.columns != 'Date'])
        test_set.loc[:, test_set.columns != 'Date'] = self.standard_scaler.inverse_transform(test_set.loc[:, test_set.columns != 'Date'])
        
        # add future values of other features
        train_preds_df = train_set[-len(train_preds):].copy()
        test_preds_df = test_set.copy()
        train_preds_df['Consumption'] = train_preds
        test_preds_df['Consumption'] = test_preds

        # rescale
        train_preds = self.standard_scaler.inverse_transform(train_preds_df.loc[:, train_preds_df.columns != 'Date'])
        test_preds = self.standard_scaler.inverse_transform(test_preds_df.loc[:, test_preds_df.columns != 'Date'])

        # Remove virtual points
        if 'virtual' in self.preprocesses:
            train_set = self.virtual_points(train_set, self.n, inverse=True)
            test_set = self.virtual_points(test_set, self.n, inverse=True)
            train_preds = self.virtual_points(train_preds, self.n, inverse=True)
            test_preds = self.virtual_points(test_preds, self.n, inverse=True)

            test_forecast_dates = self.virtual_points(test_forecast_dates, self.n, inverse=True)
            train_dates = self.virtual_points(train_dates, self.n, inverse=True)

        # Create a DataFrame of combined training set predictions and test set forecast with ground truth
        df_train = pd.DataFrame({'ds': train_dates, 'gt': train_set.iloc[-len(train_preds):]['Consumption'],
                                 'model': train_preds[:,consumption_idx]})
        df_test = pd.DataFrame({'ds': test_forecast_dates.tolist(), 'gt': test_set['Consumption'].tolist(),
                                 'forecast': test_forecast_df['Consumption'].tolist(), 'test_pred': test_preds[:,consumption_idx].tolist()})
        df_forecast = pd.concat([df_train, df_test]).reset_index(drop=True)

        # Compute evaluation metrics for the forecast
        test_metrics, forecast_df = self.evaluate_forecast(df_forecast, save_dir=save_dir, plot=plot)
        return test_metrics, forecast_df


    def forecast(self, days, test_set, recent_data=None):
        '''
        Create a forecast for the test set. Note that this is different than obtaining predictions for the test set.
        The model makes a prediction for the provided example, then uses the result for the next prediction.
        Repeat this process for a specified number of days.
        :param days: Number of days into the future to produce a forecast for
        :param recent_data: A factual example for the first prediction
        :return: An array of predictions
        '''
        if recent_data is None:
            raise Exception('RNNs require an input of shape (T_x, features) to initiate forecasting.')
        preds = np.zeros((days, 1))
        x = recent_data

        for i in range(days):
            if 'fragments' in self.preprocesses: 
                _, x_i, _ = self.fragment(pd.concat([x, test_set[i:i+1]]))
                preds[i] = self.model.predict(np.expand_dims(x_i[0], axis=0))
                x = x.shift(-1)
                new_row = test_set[i:i+1].copy()
                new_row.loc[:, 'Consumption'] = preds[i]
                x = pd.concat([x[:-1], new_row])
                
            if 'windowed' in self.preprocesses: 
                x_i = x
                preds[i] = self.model.predict(np.expand_dims(x_i, axis=0))
                x = np.roll(x, -1, axis=0)
                x[-1] = preds[i]    # Prediction becomes latest data point in the example

        preds_df = test_set.copy()
        preds_df['Consumption'] = preds

        preds = self.standard_scaler.inverse_transform(preds_df.loc[:, preds_df.columns != 'Date'])
        forecast_dates = pd.date_range(self.forecast_start, periods=days).tolist()
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Consumption': preds[:,0].tolist()})
        if 'virtual' in self.preprocesses:
            forecast_df = self.virtual_points(forecast_df, self.n, inverse=True)

        return forecast_df


    def save(self, save_dir, scaler_dir=None):
        '''
        Saves the model to disk
        :param save_dir: Directory in which to save the model
        '''
        if self.model:
            model_path = os.path.join(save_dir, self.name + self.train_date + '.h5')
            save_model(self.model, model_path)  # Save the model's weights
            dump(self.standard_scaler, scaler_dir + 'standard_scaler.joblib')


    def load(self, model_path, scaler_path=None):
        '''
        Loads the model from disk
        :param model_path: Path to saved model
        '''
        if os.path.splitext(model_path)[1] != '.h5':
            raise Exception('Model file path for ' + self.name + ' must have ".h5" extension.')
        if scaler_path is None:
            raise Exception('Missing a path to a serialized standard scaler.')
        self.model = load_model(model_path, compile=False)
        self.standard_scaler = load(scaler_path)
        return

    ### PREPROCESSING

    def preprocess(self, train_set, test_set): 
        '''
        Make time series datasets with chosen methods
        :param dataset: Pandas DataFrame indexed by date
        :return: A windowed time series dataset of shape (# rows, no inputs per feature, # features)
        '''

        # adding virtual data points
        if 'virtual' in self.preprocesses: 
            # number of virtual points to add
            self.n = 1
            split_idx = (self.n+1)*len(train_set)
            data_set = self.virtual_points(pd.concat([train_set, test_set]), n=self.n)
            train_set = data_set[:split_idx]
            test_set = data_set[split_idx:]

        # adding clusters to demand
        if 'cluster' in self.preprocesses: 
            split_idx = (self.n+1)*len(train_set)
            data_set = self.cluster(pd.concat([train_set, test_set]))
            train_set = data_set[:split_idx]
            test_set = data_set[split_idx:]

        if 'pca' in self.preprocesses: 
            split_idx = (self.n+1)*len(train_set)
            data_set = self.PCA_features(pd.concat([train_set, test_set]))
            train_set = data_set[:split_idx]
            test_set = data_set[split_idx:]

        # data selection methods
        if 'fragments' in self.preprocesses: 
            train_dates, X_train, Y_train = self.fragment(train_set)
            test_pred_dates, X_test, Y_test = self.fragment(pd.concat([train_set[len(Y_train):], test_set]))
        elif 'windowed' in self.preprocesses:
            train_dates, X_train, Y_train = self.make_windowed_dataset(train_set)
            test_pred_dates, X_test, Y_test = self.make_windowed_dataset(pd.concat([train_set[-self.T_x:], test_set]))
            
        return train_set, test_set, train_dates, X_train, Y_train, test_pred_dates, X_test, Y_test


    def make_windowed_dataset(self, dataset):
        '''
        Make time series datasets. Each example is a window of the last T_x data points and label is data point 1 day
        into the future.
        :param dataset: Pandas DataFrame indexed by date
        :return: A windowed time series dataset of shape (# rows, T_x, # features)
        '''
        if 'virtual' in self.preprocesses: 
            self.T_x = self.T_x * 2

        dates = dataset['Date'][self.T_x:].tolist()
        unindexed_dataset = dataset.loc[:, dataset.columns != 'Date']
        X = np.zeros((unindexed_dataset.shape[0] - self.T_x, self.T_x, unindexed_dataset.shape[1]))
        Y = unindexed_dataset['Consumption'][self.T_x:].to_numpy()
        for i in range(X.shape[0]):
            X[i] = unindexed_dataset[i:i+self.T_x].to_numpy()
        return dates, X, Y
    

    def cluster(self, dataset): 
        """
        Add clustering based on distance between data points
        :param cfg: project config
        :param dataset: Dataframe with raw data for each dma with no gaps and datetime index
        """
        # define number of clusters
        cluster_count = math.ceil(math.sqrt(168))

        # cluster data points
        kmeans = KMeans(n_clusters=cluster_count,max_iter=5000)
        labels = kmeans.fit_predict(dataset['Consumption'])
        df_cluster = pd.DataFrame(zip(dataset['Consumption'].flatten(),labels),columns=["data","Cluster"]).sort_values(by="Cluster")
        for cluster in df_cluster['Cluster'].unique(): 
            df_cluster['Cluster_'+str(cluster)] = 0
            df_cluster.loc[df_cluster['Cluster']==cluster, 'Cluster_'+str(cluster)] = 1
        
        df_cluster = df_cluster.drop(columns={'Cluster'})

        return df_cluster
    

    def PCA_features(self, dataset, components=None): 
        """
        Collates exogenous features into primary components using pca
        :param dataset: Dataframe with raw data for each dma with no gaps and datetime index
        :param components: Number of principal components to include
        """
        # initiate pca df
        df_pca = dataset[['Consumption']]

        # apply pca
        pca = PCA(0.95)
        pca.fit(dataset.loc[:, dataset.columns not in ['date_time', 'Consumption']])
        train_img = pca.transform(dataset.loc[:, dataset.columns not in ['date_time', 'Consumption']])

        if components == None: 
            df_pca[[str(i) for i in range(train_img.shape[1])]] = train_img
        else: 
            df_pca[[str(i) for i in range(components)]] = train_img[:, :components]

        return df_pca
    

    def virtual_points(self, dataset, n, inverse=False): 
        '''
        Introduce virtual datapoints to reduce non-linearity. 
        :param dataset: Pandas DataFrame indexed by date
        :param n: Number of virtual data points to introduce between real data
        :param inverse: set to True to reverse virtual point addittion
        :return: dataset with virtual data points via linear interpolation
        '''
        if not inverse: 
            dataset = dataset.set_index('Date')
            frequency = str(60/(n+1)) + 'min'
            dataset = dataset.asfreq(freq=frequency).interpolate()
            dataset = dataset.reset_index()
        else: 
            if isinstance(dataset, np.ndarray) or isinstance(dataset, list):
                dataset = dataset[0::(n+1)]
            else: 
                dataset = dataset.loc[dataset.reset_index().index % (n+1) == 0]

        return dataset
    

    def fragment(self, dataset, n_steps_out=1, n_steps_present=16, n_steps_recent=16, n_steps_distant=16):
        '''
        Introduce virtual datapoints to reduce non-linearity. 
        :param dataset: Pandas DataFrame indexed by date
        :param n_steps_present: Number of hourly increments to include in the present fragment window
        :param n_steps_recent: Number of hourly increments to include in the recent fragment window
        :param n_steps_distant: Number of hourly increments to include in the past fragment window
        :param n_steps_out: Forecast horizon 
        : return: dataset with virtual data points via linear interpolation
        '''
        dataset = dataset.reset_index(drop=True)
        sequence = dataset.loc[:, dataset.columns != 'Date']

        # number of hours in past for each fragment
        recent_dist = 24 
        distant_dist = 24*7

        if 'virtual' in self.preprocesses: 
            n_steps_present = n_steps_present * (self.n+1)
            n_steps_recent = n_steps_recent * (self.n+1)
            n_steps_distant = n_steps_distant * (self.n+1)
            recent_dist = recent_dist * (self.n+1)
            distant_dist = distant_dist * (self.n+1)

        n_steps_in = n_steps_present + n_steps_recent + n_steps_distant
        x = []
        Y = []
        dates = []

        #n_steps_in = n_steps_present + n_steps_recent + n_steps_distant
        for i in range(len(sequence)):
            # check if enough steps remaining for both present window and label
            if i+n_steps_present+n_steps_out > len(sequence):
                break
            
            # check if enough steps in past to form recent and distant fragments
            if ((i-recent_dist+int(n_steps_recent/2)<0) | (i-distant_dist+int(n_steps_distant/2)<0)):
                continue

            range_present = range(i,i+n_steps_present)
            range_recent = range(i,i+n_steps_recent)
            range_distant = range(i,i+n_steps_distant)

            range_predict = range(i+n_steps_present,i+n_steps_present+n_steps_out)
            
            r = range_present
            input_present = sequence[r[0]:r[-1]+1] #Data 5 timesteps before
            #input_present_e = np.concatenate((dow[i],holiday[i],time[i]),axis=None)
            #input_present_e = np.concatenate((np.concatenate(dow[r]),holiday[r],time[r]),axis=None)

            r = [r-recent_dist+int(n_steps_recent/2) for r in range_recent]
            input_recent =  sequence[r[0]:r[-1]+1] #Data one day before
            #input_recent_e = np.concatenate((np.concatenate(dow[r]),holiday[r],time[r]),axis=None)

            r = [r-distant_dist+int(n_steps_distant/2) for r in range_distant]
            input_distant =  sequence[r[0]:r[-1]+1] #Data 4days before
            #input_distant_e = np.concatenate((np.concatenate(dow[r]),holiday[r],time[r]),axis=None)
            
            input = np.concatenate((input_distant, input_recent, input_present),axis=0)
            #input = np.concatenate((input, input_present_e),axis=None)
            #input = np.concatenate((input, input_present_e, input_recent_e, input_distant_e),axis=None)
            output = sequence['Consumption'][range_predict]
                
            x.append(input)
            Y.append(output)
            dates.append(dataset['Date'][range_predict])
           
        X = np.zeros((len(x), n_steps_in, sequence.shape[1]))
        for i in range(X.shape[0]):
            X[i] = x[i]

        return dates, X, np.squeeze(np.array(Y))
    

    def get_recent_data(self, dataset):
        '''
        Given a preprocessed dataset, get the most recent factual example
        :param dataset: A DataFrame representing a preprocessed dataset
        :return: Most recent factual example
        '''
        if self.univariate:
            dataset = dataset[['Date', 'Consumption']]
        dataset.loc[:, dataset.columns != 'Date'] = self.standard_scaler.transform(dataset.loc[:, dataset.columns != 'Date'])
        test_pred_dates, X, Y = self.make_windowed_dataset(dataset[-self.T_x:])
        self.forecast_start = test_pred_dates[-1]
        return X[-1]


class ANNModel(NNModel):
    '''
    A class representing a simple neural network model with Dense layers
    '''

    def __init__(self, hparams, log_dir=None):
        name = 'ANN'
        self.dropout = hparams.get('DROPOUT', 0.25)
        self.preprocesses = hparams.get('PREPROCESS', ['fragments'])
        self.fc0_units = int(hparams.get('FC0_UNITS', 32))
        self.fc1_units = int(hparams.get('FC1_UNITS', 32))
        self.fc2_units = int(hparams.get('FC2_UNITS', 32))
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(ANNModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim)
        X = Dense(self.fc0_units, activation='relu', name='fc0')(X_input)
        X = Flatten()(X)
        if self.fc1_units is not None:
            X = Dense(self.fc1_units, activation='relu', name='fc1')(X)
            X = Dropout(self.dropout)(X)
            if self.fc2_units is not None:
                X = Dense(self.fc2_units, activation='relu', name='fc2')(X)
                X = Dropout(self.dropout)(X)
        Y = Dense(1, activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                    initial_learning_rate=self.lr,
                    decay_steps=45*1000,
                    decay_rate=1,
                    staircase=False),
                    clipvalue=0.5,
                    clipnorm=1)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model
    

class LSTMModel(NNModel):
    '''
    A class representing a recurrent neural network model with a single LSTM layer
    '''

    def __init__(self, hparams, log_dir=None):
        name = 'LSTM'
        self.units = int(hparams.get('UNITS', 128))
        self.preprocesses = hparams.get('PREPROCESS', ['fragments'])
        self.dropout = hparams.get('DROPOUT', 0.25)
        self.fc0_units = int(hparams.get('FC0_UNITS', 32))
        self.fc1_units = int(hparams.get('FC1_UNITS', None))
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(LSTMModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim)
        X = LSTM(self.units, activation='tanh', return_sequences=True, name='lstm')(X_input)
        X = LSTM(self.units, activation='tanh', name='lstm_1')(X)
        X = Flatten()(X)
        if self.fc0_units is not None:
            #X = LSTM(self.units, activation='tanh', return_sequences=True, name='lstm')
            X = Dense(self.fc0_units, activation='relu', name='fc0')(X)
            X = Dropout(self.dropout)(X)
            if self.fc1_units is not None:
                #X = LSTM(self.units, activation='tanh', return_sequences=True, name='lstm')
                X = Dense(self.fc1_units, activation='relu', name='fc1')(X)
                X = Dropout(self.dropout)(X)
        Y = Dense(1, activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                    initial_learning_rate=self.lr,
                    decay_steps=45*1000,
                    decay_rate=1,
                    staircase=False),
                    clipvalue=0.5,
                    clipnorm=1)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model


class GRUModel(NNModel):
    '''
    A class representing a recurrent neural network model with a single GRU layer
    '''

    def __init__(self, hparams, log_dir=None):
        name = 'GRU'
        self.units = int(hparams.get('UNITS', 128))
        self.preprocesses = hparams.get('PREPROCESS', ['fragments'])
        self.dropout = hparams.get('DROPOUT', 0.25)
        self.fc0_units = int(hparams.get('FC0_UNITS', [32]))
        self.fc1_units = int(hparams.get('FC1_UNITS', None))
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(GRUModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim)
        X = GRU(self.units, activation='tanh', return_sequences=True, name='gru')(X_input)
        X = GRU(self.units, activation='tanh', name='gru_1')(X)
        #X = Flatten()(X)
        if self.fc0_units is not None:
            X = Dropout(self.dropout)(X)
            X = Dense(self.fc0_units, activation='relu', name='fc0')(X)
            if self.fc1_units is not None:
                X = Dropout(self.dropout)(X)
                X = Dense(self.fc1_units, activation='relu', name='fc1')(X)
        Y = Dense(1, activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model


class CNN1DModel(NNModel):
    '''
    A class representing a 1D convolutional neural network model with a single 1D convolutional layer
    '''

    def __init__(self, hparams, log_dir=None):
        name = '1DCNN'
        self.init_filters = int(hparams.get('FILTERS', 128))
        self.filter_multiplier = int(hparams.get('FILTER_MULTIPLIER', 2))
        self.kernel_size = int(hparams.get('KERNEL_SIZE', 3))
        self.stride = int(hparams.get('STRIDE', 2))
        self.n_conv_layers = int(hparams.get('N_CONV_LAYERS', 2))
        self.fc0_units = int(hparams.get('FC0_UNITS', 32))
        self.fc1_units = int(hparams.get('FC1_UNITS', 16))
        self.dropout = hparams.get('DROPOUT', 0.25)
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(CNN1DModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim, name='input')
        X = X_input
        for i in range(self.n_conv_layers):
            try:
                X = Conv1D(self.init_filters * self.filter_multiplier**i, self.kernel_size, strides=self.stride,
                           activation='relu', name='conv' + str(i))(X)
            except Exception as e:
                print("Model cannot be defined with above hyperparameters", e)
        X = Flatten()(X)
        if self.fc0_units is not None:
            X = Dropout(self.dropout)(X)
            X = Dense(self.fc0_units, activation='relu', name='fc0')(X)
            if self.fc1_units is not None:
                X = Dropout(self.dropout)(X)
                X = Dense(self.fc1_units, activation='relu', name='fc1')(X)
        Y = Dense(input_dim[1], activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model
