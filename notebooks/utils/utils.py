# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def plot_results(forecast, test): 
    """
    plot resutls of trained model
    """
    fig, axs = plt.subplots(5,2, figsize=(8,24))
    axs = axs.flatten()

    for i, dma in enumerate(test.columns): 
        # plot results
        axs[i].plot(forecast[dma], color='orange', label='forecast', linewidth=2)
        axs[i].plot(test[dma], color='blue', label='observed', linewidth=2)
        axs[i].fill_between(forecast.index, forecast['lower_'+dma], forecast['upper_'+dma] ,color='green',alpha=0.2)

        axs[i].set_title(dma)
        axs[i].set_ylabel('demand')
        axs[i].set_xlabel('date')
        leg = axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=False, shadow=False, ncol=6, edgecolor='w', borderpad=0.5)
        leg.get_frame().set_linewidth(0.5)
    
    plt.show()

def extract_features(df): 
    """
    extract time features of data
    """
    # extract time features
    df['year'] = [d.year for d in df.index]
    df['month'] = [d.month for d in df.index]
    df['day'] = [d.weekday for d in df.index]
    df['date'] = [d.date() for d in df.index]
    df['time'] = [d.time() for d in df.index]
    df['hour'] = [d.hour() for d in df.index]

    # make list of special days for the DMAs region
    official_holidays = ["2021-01-01","2021-01-06","2021-04-04","2021-04-05","2021-04-25","2021-05-01","2021-06-02","2021-08-15","2021-11-01","2021-12-05","2021-12-25","2021-12-26","2022-01-01","2022-01-06","2022-04-17","2022-04-18","2022-04-25","2022-05-01","2022-06-22","2022-08-15","2022-11-01","2022-12-08","2022-12-25","2022-12-26"]

    legally_not_recongnized_holidays = ["2021-04-23","2021-05-23","2022-04-23","2022-06-05"]

    event_day = ["2021-03-28","2021-05-09","2021-10-31","2021-11-28","2021-05-12","2021-12-12","2021-12-19","2021-12-31","2022-03-27","2022-04-10","2022-05-08","2022-05-09","2022-10-30","2022-11-27","2022-12-04","2022-12-11","2022-12-18","2022-12-31"]

    # make columns for special days
    df['official_holiday'] = 0
    df['legally_not_recongnized_holidays'] = 0
    df['event_day'] = 0
    df['weekend'] = 0

    # add indicator variable for special days
    for i in df.index:
        if str(i)[:10] in official_holidays:
            df['official_holiday'][i] = 1

    for i in df.index:
        if str(i)[:10] in legally_not_recongnized_holidays:
            df['legally_not_recongnized_holidays'][i] = 1

    for i in df.index:
        if str(i)[:10] in event_day:
            df['event_day'][i] = 1

    # add variable for weekend days
    for i in df.index:
        if i.weekday() == 5 or i.weekday() == 6:
            df['weekend'][i] = 1

    #Vector for days
    day_arr = []
    for i in range(0,len(df)):
        day_vec = np.array([0,0,0,0,0,0,0])
        day_vec[df['day'].iloc[i]] = 1
        day_arr.append(day_vec)
    df['day_arr']=day_arr

    return df

def virtual_data(df, p): 
    """
    inserts virtual data points into input data to reduce non-linearity. 

    df - input data including several dmas
    p - number of virtual data points to include between consecutive real values
    """

    copy = df.copy()
    freq = 60/(p+1)
    resampled = copy.resample(str(freq)+'min')
    interpolated = resampled.interpolate(method='linear')   

    return interpolated        

def classify_values(df, k):
    """
    classify each data point for dma into k classes. 

    df - input data including several dmas
    k - number of classes to use
    """

    # Initialize an empty DataFrame to store the classified values
    classified_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Loop through each column in the DataFrame
    for col in df.columns:
        # Extract column values
        column_values = df[col].values.reshape(-1, 1)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(column_values)

        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Classify each value into k classes based on the cluster centers
        classified_values = np.argmin(np.abs(column_values - cluster_centers.T), axis=1)

        # Store the classified values in the classified DataFrame
        classified_df[col] = classified_values

    return classified_df

def fragment_dma(df, dma, n_steps_present=30, n_steps_recent=30, n_steps_distant=30, n_steps_out=24): 
    """
    split data into fragments to be used in neural network. 

    df - input data including several dmas
    dma - dma to produce result for
    n_steps_present - number of steps backward from forecast at t
    n_steps_recent - number of steps included from same time in previous day
    n_steps_distant - number of steps included from same time in 4th day prior
    n_steps_out - number of steps included in the forecast
    """
    sequence = df[dma].values.astype('float32')
    dow = df['day_arr'].values
    holiday = df['official_holiday'].values.astype('int')
    time = df.index.strftime("%H%M").astype('int')/2400
    
    x, y = [],[]
    #n_steps_in = n_steps_present + n_steps_recent + n_steps_distant
    for i in range(len(sequence)):
        if i+n_steps_present+n_steps_out > len(sequence):
            #Prediction out of sequence
            break
        
        if ((i-96+int(n_steps_recent/2)<0)| (i-96*4+int(n_steps_distant/2)<0)):
        #Past data not available
            continue

        range_present = range(i,i+n_steps_present)
        range_recent = range(i,i+n_steps_recent)
        range_distant = range(i,i+n_steps_distant)

        range_predict = range(i+n_steps_present,i+n_steps_present+n_steps_out)
        
        r = range_present
        input_present = sequence[r] #Data 5 timesteps before
        input_present_e = np.concatenate((dow[i],holiday[i],time[i]),axis=None)
        #input_present_e = np.concatenate((np.concatenate(dow[r]),holiday[r],time[r]),axis=None)

        r = [r-96+int(n_steps_recent/2) for r in range_recent]
        input_recent =  sequence[r] #Data one day before
        input_recent_e = np.concatenate((np.concatenate(dow[r]),holiday[r],time[r]),axis=None)

        r = [r-96*4+int(n_steps_distant/2) for r in range_distant]
        input_distant =  sequence[r] #Data 4days before
        input_distant_e = np.concatenate((np.concatenate(dow[r]),holiday[r],time[r]),axis=None)

        input = np.concatenate((input_present, input_recent, input_distant),axis=None)
        input = np.concatenate((input, input_present_e),axis=None)
        #input = np.concatenate((input, input_present_e, input_recent_e, input_distant_e),axis=None)
        output = sequence[range_predict]

        x.append(input)
        y.append(output)

    return np.squeeze(np.array(x)), np.squeeze(np.array(y))