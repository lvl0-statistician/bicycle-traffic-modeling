# Modeling bicycle traffic

### Goal
The goal of this study is to produce a model that is able to predict the daily number of bicycles passing through a sensor from midnight to 09:00 AM on April 2nd. The sensor is placed in Albert 1st totem.Towards that aim, we are given a data set consisting of the totem’s daily snapshots for over a one year period.Multiple snapshots can be taken during the same day.


### Methodology
First step is to pre-process the data by extracting a time series that is consistent with our goal. Smoothing techniques are then used to deal with the non-stationnarity of the resulting time series. An auto-regressive model AR(p) of order p is then trained on a training set where the order of the AR model is used as a design parameter. To estimate the models parameters, the normal equation from linear regression was used.

Please see 'bike_prediction.pdf' for further details.

### Results

R^2 scores are then reported on the testing set and are used to select the best auto-regressive order. The optimal order found is p=9.
The predicted number of bikes passing through the sensor between 00:00 AM and 9:00 AM is 307 bicycles.

