from community_area import *
from MCMC import *
from q_learning import *
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICE"] = "1"

'''
For more details on baseline implementation, please refer to the paper: Learning Task-Specific City Region Partition
'''
def tract_reg():
    trts = Tract.createAllTracts()
    trts_features = Tract.generateFeatures()
    default_house_price_train = trts_features['train_price'].sum() / trts_features['train_count'].sum()
    default_house_price_test = trts_features['test_price'].sum() / trts_features['test_count'].sum()
    trts_features['train_average_price'] = trts_features['train_price'] / (trts_features['train_count'])
    trts_features['test_average_price'] = trts_features['test_price'] / (trts_features['test_count'])
    trts_features['train_average_price'] = trts_features['train_average_price'].replace([np.inf, -np.inf, np.nan], default_house_price_train)
    trts_features['test_average_price'] = trts_features['test_average_price'].replace([np.inf, -np.inf, np.nan], default_house_price_test)
    # print(trts_features)
    crime_error_train = Linear_regression_evaluation(trts_features, 'train_crime')
    price_error_train = Linear_regression_evaluation(trts_features, 'train_average_price')
    crime_error_test = Linear_regression_evaluation(trts_features, 'test_crime')
    price_error_test = Linear_regression_evaluation(trts_features, 'test_average_price')
    origin_tract_reg = pd.DataFrame(data=None, columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    origin_tract_reg.loc['crime_error_train'] = crime_error_train
    origin_tract_reg.loc['price_error_train'] = price_error_train
    origin_tract_reg.loc['crime_error_test'] = crime_error_test
    origin_tract_reg.loc['price_error_test'] = price_error_test

    print(origin_tract_reg)

    # origin_tract_reg.to_csv("output/origin_tract_reg/origin_tract_regression.csv")
    origin_tract_reg.to_csv("./output/preds/origin_tract_regression.csv")



def baseline_clustering(n_simu):
    tasks = ['crime', 'house-price']
    for task in tasks:
        results, means, std = fig_clustering_baseline2(task=task, n_sim=n_simu, cluster_X=True, cluster_y=True)

        means.to_csv("./output/baseline_clustering/baselines-mean-results-{}-{}.csv".format('based_XY', task))
        std.to_csv("./output/baseline_clustering/baselines-std-results-{}-{}.csv".format('based_XY', task))

        pd.DataFrame(results[:, 0, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals']).to_csv(
            "./output/baseline_clustering/baseline-clustering-{}-{}-results.csv".format(task, 'admin'))
        pd.DataFrame(results[:, 1, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals']).to_csv(
            "./output/baseline_clustering/baseline-clustering-{}-{}-results.csv".format(task, 'kmeans'))
        pd.DataFrame(results[:, 2, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals']).to_csv(
            "./output/baseline_clustering/baseline-clustering-{}-{}-results.csv".format(task, 'agg'))
        pd.DataFrame(results[:, 3, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals']).to_csv(
            "./output/baseline_clustering/baseline-clustering-{}-{}-results.csv".format(task, 'spectral'))


def baseline_mcmc(n_simu):
    random.seed(100)
    results = np.zeros(shape=(n_simu, 4, 5))
    for i in range(n_simu):
        version = "v{}".format(i+1)
        print("-----{}-----".format(version))
        # Crime
        crime_naive_reg = naive_MCMC('crime-naive-{}'.format(version), targetName='train_crime', lmbda=0.03, f_sd=0.008, Tt=1)
        crime_soft_reg = MCMC_softmax_proposal('crime-softmax-{}'.format(version), targetName='train_crime', lmbda=0.03, f_sd=0.008, Tt=1)
        # House Prices
        price_naive_reg = naive_MCMC('house-price-naive-{}'.format(version), targetName='train_average_house_price', lmbda=0.0004, f_sd=0.008, Tt=1)
        price_soft_reg = MCMC_softmax_proposal('house-price-softmax-{}'.format(version), targetName='train_average_house_price', lmbda=0.0004, f_sd=0.008, Tt=1)

        results[i, 0, :] = crime_naive_reg
        results[i, 1, :] = crime_soft_reg
        results[i, 2, :] = price_naive_reg
        results[i, 3, :] = price_soft_reg

    crime_naive_simu = pd.DataFrame(results[:, 0, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    crime_soft_simu = pd.DataFrame(results[:, 1, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    price_naive_simu = pd.DataFrame(results[:, 2, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    price_soft_simu = pd.DataFrame(results[:, 3, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])

    crime_naive_simu.to_csv("./output/baseline_mcmc/{}-mcmc-results.csv".format('crime_naive_simu'))
    crime_soft_simu.to_csv("./output/baseline_mcmc/{}-mcmc-results.csv".format('crime_soft_simu'))
    price_naive_simu.to_csv("./output/baseline_mcmc/{}-mcmc-results.csv".format('price_naive_simu'))
    price_soft_simu.to_csv("./output/baseline_mcmc/{}-mcmc-results.csv".format('price_soft_simu'))

    means = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    print(means)
    print(std)

    means = pd.DataFrame(means, index=['crime_naive', 'crime_soft', 'price_naive', 'price_soft'], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    std = pd.DataFrame(std, index=['crime_naive', 'crime_soft', 'price_naive', 'price_soft'], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    means.to_csv("./output/baseline_mcmc/baselines-mean-results-{}.csv".format('mcmc'))
    std.to_csv("./output/baseline_mcmc/baselines-std-results-{}.csv".format('mcmc'))



def baseline_qlearning(n_simu):
    # random.seed(10)
    results = np.zeros((n_simu, 2, 5))
    for v in range(n_simu):
        crime_dqn = q_learning('crime-q-learning-maxiter-test-v{}'.format(v + 1), targetName='train_crime', lmbd=0.0004, f_sd=0.005, Tt=1)
        price_dqn = q_learning('house-price-q-learning-sampler-v{}'.format(v + 1), targetName='train_average_house_price', lmbd=0.005, f_sd=3, Tt=0.1)
        results[v, 0, :] = crime_dqn
        results[v, 1, :] = price_dqn

    crime_dqn_simu = pd.DataFrame(results[:, 0, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    price_dqn_simu = pd.DataFrame(results[:, 1, :], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    crime_dqn_simu.to_csv("./output/baseline_dqn/{}-results.csv".format('crime_dqn_simu'))
    price_dqn_simu.to_csv("./output/baseline_dqn/{}-results.csv".format('price_dqn_simu'))

    means = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    print(means)
    means = pd.DataFrame(means, index=['crime_dqn', 'price_dqn'], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    std = pd.DataFrame(std, index=['crime_dqn', 'price_dqn'], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    means.to_csv("./output/baseline_dqn/baselines-mean-results-{}.csv".format('dqn'))
    std.to_csv("./output/baseline_dqn/baselines-std-results-{}.csv".format('dqn'))



if __name__ == '__main__':
    n_simu = 10
    # tract_reg()
    baseline_clustering(n_simu)
    # baseline_mcmc(n_simu)
    baseline_qlearning(n_simu)

