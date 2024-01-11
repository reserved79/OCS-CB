from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier, Pool
from OBL_CS import obl_cuckoo_search

def ocs_cb(X_train, X_validation, Y_train, Y_validation):
    train_pool = Pool(X_train, label=Y_train)
    test_pool = Pool(X_validation, label=Y_validation)

    def fit_func(p):
        train_pool = Pool(X_train, label=Y_train)
        test_pool = Pool(X_validation, label=Y_validation)

        params = {
            'learning_rate': 0.01 + (1 - 0.01) * p[0],
            'num_boost_round': int(10 + (1000 - 10) * p[1]),
            'max_depth': int(3 + (10 - 3) * p[2]),
            'subsample': 0.5 + (1 - 0.5) * p[3],
            'reg_lambda': int(1 + (100 - 1) * p[4])
        }
        model = CatBoostClassifier(**params, verbose=False)
        model.fit(train_pool, eval_set=test_pool)
        result = cross_val_score(model, X_train, Y_train, cv=5, scoring='recall')
        fitness = -result.mean()
        return fitness


    for i in range(30):
        best_nest, best_fitness, best_t1 = obl_cuckoo_search(30, 5, fit_func, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1],
                                                         step_size=0.4)
        print((best_fitness, best_nest))
        # print(best_t1)