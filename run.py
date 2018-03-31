import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

sample = pd.read_csv("./sample.csv")
print(sample)


features = sample[['1', '2']].as_matrix()
targets = sample['3'].as_matrix()


def mlp_regressor_grid_search():
    est = MLPRegressor(activation= 'logistic')
    sizes = []
    for size1 in range(5, 20, 5):
        sizes.append(size1)
        for size2 in range(5, 20, 5):
            sizes.append((size1, size2))
            # for size3 in range(10,30,10):
            #     sizes.append((size1, size2, size3))
    param_grid = dict(solver=['lbfgs'], #'sgd','adam'
                      learning_rate=['adaptive','invscaling'],
                      alpha=[0.0001], # [0.0005, 0.001, 0.005, 0.01]
                      max_iter=[200], #np.arange(200, 300, 20)
                      tol=[0.0001], #np.arange(0.00001, 0.0001, 0.00001)
                      hidden_layer_sizes=sizes)
    return GridSearchCV(est, param_grid=param_grid, n_jobs=1, verbose=100)


def split_by_position(features, targets):
    """
    train 0.80
    test 0.20
    """
    len_train = int(0.80 * len(features))
    train_features = features[0:len_train]
    train_targets = targets[0:len_train]
    test_features = features[len_train:]
    test_targets = targets[len_train:]
    return train_features, test_features, train_targets, test_targets


train_features, test_features, train_targets, test_targets = split_by_position(features, targets)
print(features)
print(targets)


scaler = StandardScaler()
scaler.fit(train_features)

scaled_train_features = scaler.transform(train_features)

est = mlp_regressor_grid_search()

est.fit(scaled_train_features, train_targets)

print(est.best_estimator_)

print(est.score(scaler.transform(test_features), test_targets))

predict_data = [[5,4]]
scaled_predict = scaler.transform(predict_data)


prediction = est.predict(scaled_predict)
print("PREDICTION")
print(prediction)