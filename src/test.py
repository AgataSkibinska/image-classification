from notebooks.utils import *
from predict import predict
from sklearn.metrics import classification_report

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data("../data/training_data/train.pkl")

    y_pred = predict(X_test)
    print(classification_report(y_test, y_pred))
