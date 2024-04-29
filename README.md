# WaterDemandForecasting
Analysis of different water demand forecasting techniques for final year project. 

## To Do

> ### 1. TTSA
* Finalise 3 models to test
- SARIMAX 24Hr [Done]
- SARIMAX 168Hr  [Long Train]
- TBATS 24/168Hr 

* Decide on amount of data in training window 
- 1 Week  
- 2 Week 
- 4 Week 
- 16 Week 

* Decide on exogenous vars used
- Temperature
- Diurnal flow

* Decide on tests to run
- SARIMAX 24Hr: 1W, 2W, 4W, 16W
- SARIMAX 168Hr: Best 2 from above
- TBATS 24Hr/168Hr: Best from above methods

* Decide on paramater selection method
- Grid search using auto arima for SARIMAX
- !Find method for TBATS!

* Decide on validation method
- Growing window cross validation
- 4 weeks test data for week ahead
- 2 weeks test data for day ahead

* Decide on forecast window
- 1 week ahead
- 1 Day ahead

* Write up results
___

> ### 2. NN

* Finalise 3 models to test
- 3 layer simple ANN
- 3 layer GRU
- 1D conv graph with GRU

* Decide on amount of data in training window
- 3 fragments based on simple ANN cross val: location and length

* Decide on exogenous vars used
- weather? 
- time features? 
- other dma? 

* Preprocessing
- fragments  [Done]
- virtual points  [Done]
- anomoly removal  [Done]
- standardization  [Done]
- clustering  [Done]
- imputing  [Done]
- PCA  [Done]
- DWT  [Done]
- features
- SARIMA residuals  [Done]

* Decide on tests to run

* Simple ANN * 
- imputing with 1. mean day
- 2. Knn
- 3. new method
- clusters as input
- PCA as input
- DWT as input
- fragments
> find best output

- apply best method to GRU network
- apply best method to 1D CGNN network and see impact
- apply GRU and 1D CGNN to SARIMAX residuals

* Decide on paramater selection method
- Bayesian hyperparam tuning

* Decide on validation method
- Growing window cross validation  [Done]
- 4 weeks test data for week ahead
- 2 weeks test data for day ahead

* Decide on forecast window
- 1 week ahead
- 1 Day ahead

* Write up results
- how forecast accuracy depends on DMA
- are pre-processing technqiues effective on cutting edge networks
- what is computational time like

- cluster dmas to find representive set so don't have to test all 10.
- highlight all gaps in intro duction but focus just on one or two.
- map of different gaps.  
