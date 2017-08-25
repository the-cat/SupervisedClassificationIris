import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from operator import itemgetter


def get_iris_dataset():
    """
    Gets the iris dataset
    :return: The iris dataset and a 2D array
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    return pandas.read_csv(url, names=names)


def get_models():
    """
    Get the models we will use to classify
    :return: A list of models
    """
    models = {}
    models['LR'] = LogisticRegression()
    models['LDA'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier()
    models['CART'] = DecisionTreeClassifier()
    models['NB'] = GaussianNB()
    models['SVM'] = SVC()
    return models


def train_model(model, X_train, Y_train, seed=7):
    """
    Train a model using 10-fold cross validation
    :param model: The model instance
    :param x_train: The training input data
    :param y_train: The training label data
    :param seed: Random seed
    :return The cross validation result - an estimation of model accuracy
    """
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    return cv_results


def main():
    dataset = get_iris_dataset()
    # separate the dataset in to X(inputs) and Y(labels)
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]

    # select 20% of the data as random validation data, the rest is training data
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)
    # train each model in turn
    models = get_models()
    results = []
    names = []
    for name, model in models.items():
        cv_results = train_model(model, X_train, Y_train)
        results.append((name, cv_results.mean(), cv_results.std()))
        names.append(name)
        msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # select the model with the highest mean accuracy score
    max_accuracy_model_name = max(results, key=itemgetter(1))[0]
    print('Max accuracy model: {}'.format(max_accuracy_model_name))
    max_accuracy_model = models[max_accuracy_model_name]

    # make some predictions using our validation data
    max_accuracy_model.fit(X_train, Y_train)
    predictions = max_accuracy_model.predict(X_validation)

    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

if __name__ == '__main__':
    main()
