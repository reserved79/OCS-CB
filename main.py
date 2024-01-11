
from data_preprocessing import read_data
from data_preprocessing import split_data
from val_cb_ass import val_cb_ass
from tra_ml_ass import tra_ml_ass
from ocs_cb import ocs_cb

if __name__ == '__main__':
    data = read_data()
    X, Y, X_train, X_validation, Y_train, Y_validation = split_data(data)
    # tra_ml_ass(X_train, Y_train)
    ocs_cb(X_train, X_validation, Y_train, Y_validation)
    # val_cb_ass(X_train, X_validation, Y_train, Y_validation)
