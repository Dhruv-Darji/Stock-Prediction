from libraries import *

def _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test):
    # Create a dictionary of our models
    estimators = [('knn', knn_model), ('rf', rf_model), ('gbt', gbt_model)]

    # Create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='hard')

    # fit model to training data
    ensemble.fit(X_train, y_train)

    # test our model on the test data
    print(ensemble.score(X_test, y_test))

    prediction = ensemble.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    return ensemble


#ensemble_model = _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test)