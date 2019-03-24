<h3>Question 1</h3>

Update values of 's' and 'D' to verify the code. 

<h3>Question 3</h3>

Candy Production data:

<img src="https://raw.githubusercontent.com/shahnazmshariff/takehome-questions/master/solutions/images/candy_production_data.png">

<h4>Basic stats on the dataset:</h4>

```
         IPG3113N
count  548.000000
mean   100.662524
std     18.052931
min     50.668900
25%     87.862475
50%    102.278550
75%    114.691900
max    139.915300 
```

<h4>Train Vs. Validation loss plot</h4>
The plot shows that the fit of the LSTM model is good as the train and validation loss decrease and stabilize towards the end. The spikes are caused due to the use of dropouts. 

<img src="https://raw.githubusercontent.com/shahnazmshariff/takehome-questions/master/solutions/images/train_vs_validation_loss.png">

<h4>Prediction results</h4>

<img src="https://raw.githubusercontent.com/shahnazmshariff/takehome-questions/master/solutions/images/prediction_result.png">

<h4>Performance stats of the model</h4>

```
Actual Bias: 1.1002550711837478
Actual RMSE: 5.180315
```

<h4>Actual values of predictions around the 90% confidence interval</h4>

<img src="https://raw.githubusercontent.com/shahnazmshariff/takehome-questions/master/solutions/images/confidence_int_90.png">

<h4>Predicting the next value in the timeseries (i.e., for Sept 2017)</h4>

```
The mean value calculated from 1000 iterations of predictions = 112.22
Uncertainty = 1.308
```


