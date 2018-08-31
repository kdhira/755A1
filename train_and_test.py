import numpy as np

from sklearn import svm, linear_model, tree, neighbors, naive_bayes

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, r2_score



def test_all(ml_type, x_train, y_train, x_test, y_test):
    clf_models = [train_svc, train_perceptron, train_dtree, train_knn, train_nbc]
    reg_models = [train_ord_reg, train_ridge_reg]
    if ml_type == 'reg':
        models = reg_models
    else:
        models = clf_models

    for model in models:
        clf = model(x_train, y_train)
        if ml_type == 'reg':
            test_reg(clf, x_test, y_test)
        else:
            test_clf(clf, x_test, y_test)

def test_clf(clf, x_test, y_test):
    print(clf.best_params_)
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))

def test_reg(clf, x_test, y_test):
    print(clf.best_params_)
    y_true, y_pred = y_test, clf.predict(x_test)
    print("Mean squared error: %.2f" % mean_squared_error(y_true, y_pred))
    print('Variance score: %.2f' % r2_score(y_true, y_pred))

def train_svc(x_train, y_train):
    svc = svm.SVC(C=1)
    svc_tp = [{
        'kernel' : ['linear'], 
        'C' : [1, 10, 100, 1000, 10000]
    }]
    svc_clf = GridSearchCV(svc, svc_tp)
    svc_clf.fit(x_train, y_train)
    return svc_clf

def train_perceptron(x_train, y_train):
    perceptron = linear_model.Perceptron()
    perceptron_tp = [{
        'max_iter' : [1000],
        'tol' : [None, 0.001, 0.0001],
        'shuffle' : [True, False],
        'eta0' : [1, 0.5, 0.25, 2, 4],
        'class_weight' : [None, 'balanced']
    }]
    perceptron_clf = GridSearchCV(perceptron, perceptron_tp)
    perceptron_clf.fit(x_train, y_train)
    return perceptron_clf

def train_dtree(x_train, y_train):
    dtree = tree.DecisionTreeClassifier()
    dtree_tp = [{
        'criterion' : ['gini', 'entropy']
    }]
    dtree_clf = GridSearchCV(dtree, dtree_tp)
    dtree_clf.fit(x_train, y_train)
    return dtree_clf

def train_knn(x_train, y_train):
    knn = neighbors.KNeighborsClassifier()
    knn_tp = [{
        'n_neighbors': np.arange(1,10)
    }]
    knn_clf = GridSearchCV(knn, knn_tp)
    knn_clf.fit(x_train, y_train)
    return knn_clf

def train_nbc(x_train, y_train):
    nbc = naive_bayes.GaussianNB()
    nbc_tp = [{
    }]
    nbc_clf = GridSearchCV(nbc, nbc_tp)
    nbc_clf.fit(x_train, y_train)
    return nbc_clf

def train_ord_reg(x_train, y_train):
    ord_reg = linear_model.LinearRegression()
    ord_reg_tp = [{
        'fit_intercept' : [True, False],
        'normalize' : [True, False],
    }]
    ord_reg_clf = GridSearchCV(ord_reg, ord_reg_tp)
    ord_reg_clf.fit(x_train, y_train)
    return ord_reg_clf

def train_ridge_reg(x_train, y_train):
    ridge_reg = linear_model.Ridge()
    ridge_reg_tp = [{
        'fit_intercept' : [True, False],
        'normalize' : [True, False],
        'tol': [0.001, 0.0001],
        'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'alpha': [1, 0.5, 0.1, 0.01, 0.001]
    }]
    ridge_reg_clf = GridSearchCV(ridge_reg, ridge_reg_tp)
    ridge_reg_clf.fit(x_train, y_train)
    return ridge_reg_clf

