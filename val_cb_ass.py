from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve


def val_cb_ass(X_train, X_validation, Y_train, Y_validation):
    # OCS-catboost

    train_pool = Pool(X_train, label=Y_train)
    test_pool = Pool(X_validation, label=Y_validation)

    # Default
    # params = {
    #
    #     'learning_rate': 0.03,
    #     'num_boost_round': 500,
    #     'max_depth': 6,
    #     'subsample': 0.66,
    #     'reg_lambda': 3,
    #     'eval_metric': 'Recall'
    # }

    # OCS_CB
    params = {

        'learning_rate': 0.77050522,
        'num_boost_round': int(524),
        'max_depth': int(5),
        'subsample': 0.89777281,
        'reg_lambda': int(58),
        'eval_metric': 'Recall'
    }

    model_cb = CatBoostClassifier(**params, verbose=False)
    model_cb.fit(train_pool, eval_set=test_pool)
    predictions_cb = model_cb.predict(X_validation)

    print('测试集')
    print('准确率（accuracy）:%s' % accuracy_score(Y_validation, predictions_cb))
    print('精准率（precision）:%s' % precision_score(Y_validation, predictions_cb))
    print('召回率（recall）:%s' % recall_score(Y_validation, predictions_cb))
    print('f1:%s' % f1_score(Y_validation, predictions_cb))
    print('FNR:%s' % FNR(Y_validation, predictions_cb))
    # print(confusion_matrix(Y_validation, predictions_cb))
    # print(classification_report(Y_validation, predictions_cb))


    # ROC曲线图
    def auc_roc_plot(X_train, X_validation, Y_train, Y_validation):
        y_pred_proba = model_cb.predict_proba(X_validation)
        fpr, tpr, thresholds = roc_curve(Y_validation, y_pred_proba[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        pyplot.figure()
        lw = 2
        pyplot.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.4f)" % roc_auc,
        )
        print('AUC:%s' % roc_auc)
        # print('fpr:%s' % fpr)
        pyplot.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.05])
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("True Positive Rate")
        pyplot.title("Receiver operating characteristic example")
        pyplot.legend(loc="lower right")
        # pyplot.savefig('auc_roc.pdf')
        pyplot.show()

    auc_roc_plot(X_train, X_validation, Y_train, Y_validation)

    precision, recall, thresholds = precision_recall_curve(Y_validation, predictions_cb)
    print('AUPR:%s' % auc(recall, precision))


def FNR(y, y_pred):
    tp = calculate_TP(y, y_pred)
    fn = calculate_FN(y, y_pred)
    return fn / (fn + tp)

def calculate_TP(y, y_pred):
    tp = 0
    for i, j in zip(y, y_pred):
        if i == j == 1:
            tp += 1
    return tp

def calculate_FN(y, y_pred):
    fn = 0
    for i, j in zip(y, y_pred):
        if i == 1 and j == 0:
            fn += 1
    return fn