# Stock Prediction

This is python and datascience project to predict stock value which will give predict stock go up or own after 15 days from today.

## How to use :

* Clone this repository to your system.
* Run the main.py file In Your device.
* Get output about whether the stock will go up or down after 15 days.

## About the dataset:

* Here in our project we import yfinanace module to get dataset of particular stock.

### yfinanace:

  * The yfinance is one of the famous modules in Python, which is used to collect online data, and with it, we can collect the financial data of Yahoo. With the help of the yfinance module, we retrieve and collect the company's financial information (such as financial ratios, etc.)
  
### How to import and use yfinanace:

```
import yfinance as yf

#downloading data from yfinanace
data = yf.download(symbol, start=start, end=end, interval=INTERVAL)
```
