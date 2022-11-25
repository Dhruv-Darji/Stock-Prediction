from sklearn.metrics import accuracy_score
from Ensemble import _ensemble_model
from KNN_model import _train_KNN
from Random_Forest import _train_random_forest
from libraries import *


"""Here We defining some Constants"""

NUM_DAYS = 10000     # The number of days of historical data to retrieve
INTERVAL = '1d'     # Sample rate of historical data
symbol = 'SPY'      # Symbol of the desired stock
# List of symbols for technical indicators
INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

"""Step 1: Now we pull the historical data using yfinance"""

start = (datetime.date.today() - datetime.timedelta( NUM_DAYS ) )
end = datetime.datetime.today()

data = yf.download(symbol, start=start, end=end, interval=INTERVAL)
data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
print(data.head())

""""Step 2: Now we will clean our data"""
def _exponential_smooth(data, alpha):
    """"here ewm means exponantial weighted function
        where alpha is smoothing function for the data"""
    return data.ewm(alpha=alpha).mean()

data = _exponential_smooth(data, 0.65)

tmp1 = data.iloc[-60:]
tmp1['close'].plot()


def cross_Validation(data):
    # Split data into equal partitions of size len_train

    num_train = 10  # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 40  # Length of each train-test set

    # Lists to store the results from each model
    rf_RESULTS = []
    knn_RESULTS = []
    ensemble_RESULTS = []

    i = 0
    while True:

        # Partition the data into chunks of size len_train every num_train days
        df = data.iloc[i * num_train: (i * num_train) + len_train]
        i += 1
        print(i * num_train, (i * num_train) + len_train)

        if len(df) < 40:
            break

        y = df['pred']
        features = [x for x in df.columns if x not in ['pred']]
        X = df[features]
        print(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=7 * len(X) // 10, shuffle=False)

        rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
        knn_model = _train_KNN(X_train, y_train, X_test, y_test)
        ensemble_model = _ensemble_model(rf_model, knn_model, X_train, y_train, X_test, y_test)

        rf_prediction = rf_model.predict(X_test)
        knn_prediction = knn_model.predict(X_test)
        ensemble_prediction = ensemble_model.predict(X_test)

        print('rf prediction is ', rf_prediction)
        print('knn prediction is ', knn_prediction)
        print('ensemble prediction is ', ensemble_prediction)
        print('truth values are ', y_test.values)

        rf_accuracy = accuracy_score(y_test.values, rf_prediction)
        knn_accuracy = accuracy_score(y_test.values, knn_prediction)
        ensemble_accuracy = accuracy_score(y_test.values, ensemble_prediction)

        print(rf_accuracy, knn_accuracy, ensemble_accuracy)
        rf_RESULTS.append(rf_accuracy)
        knn_RESULTS.append(knn_accuracy)
        ensemble_RESULTS.append(ensemble_accuracy)

    print('RF Accuracy = ' + str(sum(rf_RESULTS) / len(rf_RESULTS)))
    print('KNN Accuracy = ' + str(sum(knn_RESULTS) / len(knn_RESULTS)))
    print('Ensemble Accuracy = ' + str(sum(ensemble_RESULTS) / len(ensemble_RESULTS)))


cross_Validation(data)