"""
Perform five-fold cross-validation on the training set to test the model's performance.
"""

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from matplotlib import pyplot


def tra_ml_ass(X_train, Y_train):
    # 评估算法的基准
    num_folds = 5
    seed = 7
    scoring = 'accuracy'

    # 评估算法
    models = {}
    predictions = {}


    models['KNN'] = KNeighborsClassifier()
    models['CART'] = DecisionTreeClassifier()
    models['LR'] = LogisticRegression(solver='liblinear')
    models['SVM'] = SVC()
    models['XGB'] = XGBClassifier()
    models['CB'] = CatBoostClassifier(verbose=False)

    results = []
    for key in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        print('%s : %f(%f)' % (key, cv_results.mean(), cv_results.std()))

    # 评估算法的箱线图
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(models.keys())
    pyplot.show()