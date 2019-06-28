from sklearn.preprocessing import LabelEncoder
from read_data_1d_other_classifiers import read_data
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def run():
    X, y = read_data()
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(X, y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    y = integer_encoded.reshape(len(integer_encoded), 1)

    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = GaussianNB()
    print("Training new iteration on " + str(X_train.shape[0]) + " training samples, " + str(
        X_test.shape[0]) + " validation samples, this may be a while...")

    history = clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, np.array(y_pred))
    '''
    accs = []
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GaussianNB()
        print("Training new iteration on " + str(X_train.shape[0]) + " training samples, " + str(
            X_test.shape[0]) + " validation samples, this may be a while...")

        history = clf.fit(X_train, y_train.ravel())
        y_pred = clf.predict(X_test)
        predicted_class1 = np.zeros(y_pred.shape)
        acc = accuracy_score(y_test, np.array(y_pred))
        accs.append(acc)
        print("Accuracy " + str(index + 1) + " is: " + str(acc * 100) + "%")'''
    print("Average accuracy " + str(np.mean(acc) * 100) + "%")
    with open("Results_other_classifiers.txt", "a+") as f:
        f.write(
            "Accuracy for Gaussian Naive Bayes Classifier with 1d data without 10-fold cross validation is: " + str(
                np.mean(acc) * 100) + "%" + "\n")
    f.close()


if __name__ == "__main__":
    run()