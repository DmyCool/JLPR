import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model
import pandas as pd
# import statsmodels.api as sm

def computeError(y_truths, y_preds, metric='mse', precision=3):
    y_truths = np.array(y_truths)
    y_preds = np.array(y_preds)
    y_preds[y_preds < 0] = 0

    residual = y_truths - y_preds

    if metric == 'mse':
        error_metric = np.mean(np.power(residual, 2))
    elif metric == 'mae':
        error_metric = np.mean(np.abs(residual))
    elif metric == 'mape':
        epsilon = np.finfo(np.float64).eps
        error_metric = np.mean(np.abs(residual) / np.maximum(np.abs(y_truths), epsilon)) * 100
    elif metric == 'r2':
        error_metric = 1 - np.sum(np.power(residual, 2)) / np.sum(np.power(y_truths-np.mean(y_truths), 2))

    elif metric == 'sr':
        error_metric = np.sum(np.abs(residual))

    elif metric == 'sse':
        error_metric = np.sum(np.power(residual, 2))
    elif metric == 'sst':
        error_metric = np.sum(np.power(y_truths-np.mean(y_truths), 2))

    elif metric == 'mre' and y_truths is not None:
        error_metric = np.sum(np.abs(residual)) / np.sum(y_truths)

    else:
        raise Exception("error metric must be mse, rmse, sse, or mre")

    return np.round(error_metric, precision)




def Linear_regression_evaluation(df, targetName):
    """
    Use sklearn linear_model to evaluate LR model (with leave one out).
    """
    loo = LeaveOneOut()
    features = pd.DataFrame(np.hstack([np.array(list(df['poi'])), np.array(list(df['img']))]), index=df.index)
    task = df[targetName]
    #    standardScaler = preprocessing.StandardScaler()
    #    features = standardScaler.fit_transform(features_raw)
    y_truths = []
    y_preds = []
    residual = []
    for train_idx, test_idx in loo.split(df):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = task.iloc[train_idx], task.iloc[test_idx]
        lrmodel = linear_model.LinearRegression()
        lr_res = lrmodel.fit(X_train, y_train)
        y_pred = lr_res.predict(X_test)
        y_preds.append(y_pred[0])
        y_truths.append(y_test.iat[0])
        residual.append(y_test.iat[0] - y_pred[0])

    mse = computeError(y_truths, y_preds, metric='mse')
    rmse = np.sqrt(mse)
    mae = computeError(y_truths, y_preds, metric='mae')
    mape = computeError(y_truths, y_preds, metric='mape')
    r2 = computeError(y_truths, y_preds, metric='r2')

    residuals = np.sum(np.abs(np.array(residual)))

    # print('rmse:{}, mae:{}, mape:{}, r2:{}, sr:{}, sse:{}, num_residual:{}'.format(rmse, mae, mape, r2, sr, sse, len(residual)))
    print('rmse:{}, mae:{}, mape:{}, r2:{}, ytruth:{}, ypreds:{}'.format(rmse, mae, mape, r2, y_truths, y_preds))

    # save predicted values
    # with open('./output/preds/{}_{}.txt'.format(targetName,taged), 'w') as f:
    #     for i in range(len(y_preds)):
    #         f.write('{},{},{}\n'.format(y_preds[i], y_truths[i], abs(residual[i])))
    #     f.close()

    return rmse, mae, mape, r2, residuals



def Linear_regression_training(df, targetName):
    """
    Use sklearn linear_model to train LR model (training to find best partition)
    """
    task = df[targetName]
    # poi = pd.DataFrame(np.array(list(df['poi'])), index=df.index)
    # img = pd.DataFrame(np.array(list(df['img'])), index=df.index)
    features = pd.DataFrame(np.hstack([np.array(list(df['poi'])), np.array(list(df['img']))]), index=df.index)

    lrmodel = linear_model.LinearRegression()
    lrmodel.fit(features, task)  # coef: lr_res.coef_
    y_pred = lrmodel.predict(features)
    errors = abs(y_pred - task)
    rel_errors = (errors - np.mean(errors))/np.std(errors)
    return np.mean(errors), np.std(errors), np.mean(errors)/np.mean(task), rel_errors



if __name__ == '__main__':

    from model.tract import Tract
    from model.community_area import CommunityArea
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    # featureName = CommunityArea.featureNames
    targetName = 'train_crime'
    print(Linear_regression_training(CommunityArea.features, targetName))
    print(Linear_regression_evaluation(CommunityArea.features, targetName))


