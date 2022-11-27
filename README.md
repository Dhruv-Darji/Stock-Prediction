# Stock Prediction

Hello this is a stock prediction system through which you will be able to predict the closing value of any stock listed in BSE or NSE by giving certain set of inputs.The models which we have used gives a pretty accurate results of the closing price of the stock.

## How to use :

* Clone this repository to your system
* Run the .ipynb file of the particular stock that you want to predict in jupyter notebook.
* Give your input in the dataframe as stock ticker.
* Get output about whether a stock will go up or down.

## About the dataset:

Here in our project we had importorted dataset from yahoo finance website. 

## Steps:


### Used libraries:

* numpy
* pandas
* matplotlib
* pandas_datareader

### Loading the data in dataframe

### Preprocessing the data :
* We cleaned the data by removing the irrelevant features.
* Splitted the dataset using in train and test form using train test split.
* For LSTM model we transformed the dataset into 0 and 1 .
* We splitted 70% of data into training to train the model and 30% to test

### Data Visualization : 
* Used matplotlib  to visualize the data.
* plotted closing price,100 days closing average and 200 days closing average on graph.

### Used LSTM mode

### Predicting the data :
* We predictied the model on X_train and then tested it on Y_test.


## Credits :
### Harsh Gandhi :
* Worked on Stock prediction code.
* Used LSTM .
* Contributed in PPT and code.

### Dhruv Darji :
* Worked on Stock prediction code.
* Used LSTM .
* Contributed in PPT and code.
