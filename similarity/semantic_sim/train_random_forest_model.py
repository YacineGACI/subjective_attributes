from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error # For evaluating the trained model
import pickle

from data import read_processed_data



if __name__ == "__main__":

    # Read train & test datasets
    Xtrain, Ytrain = read_processed_data("data/processed/train.csv")
    Xtest, Ytest = read_processed_data("data/processed/test.csv")

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
    rf_model.fit(Xtrain, Ytrain)

    print("Training complete")

    # Saving the trained models
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # Evaluate the trained model
    y_pred_rf = rf_model.predict(Xtest)

    print("MSE --> ", mean_squared_error(Ytest, y_pred_rf))
    print("MAE --> ", mean_absolute_error(Ytest, y_pred_rf))