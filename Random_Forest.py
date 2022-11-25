from libraries import *

def _train_random_forest(X_train, y_train, X_test, y_test):
    """
    Function that uses random forest classifier to train the model
    :return:
    """

    # Create a new random forest classifier
    rf = RandomForestClassifier()

    # Dictionary of all values we want to test for n_estimators
    params_rf = {'n_estimators': [110, 130, 140, 150, 160, 180, 200]}

    # Use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf, cv=5)

    # Fit model to training data
    rf_gs.fit(X_train, y_train)

    # Save best model
    rf_best = rf_gs.best_estimator_

    # Check best n_estimators value
    print(rf_gs.best_params_)

    prediction = rf_best.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    return rf_best


#rf_model = _train_random_forest(X_train, y_train, X_test, y_test)