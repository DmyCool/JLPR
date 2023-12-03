from tract import Tract
from community_area import CommunityArea
from regression import Linear_regression_training, Linear_regression_evaluation
import random
import numpy as np
from shapely.ops import unary_union
from mcmcSummaries import plotMcmcDiagnostics, writeSimulationOutput
import logging
import warnings
warnings.filterwarnings("ignore")

def initialize(project_name, targetName, lmbd=0.75, f_sd=1.5, Tt=10, init_ca=True):
    global M, T, lmbda, featureName, CA_maxsize, mae1, errors1, cnt, iter_cnt, \
        mae_series, mae_index, sd_series, pop_sd_1, f_series, epsilon
    print("# initialize {}".format(project_name))
    """
    epsilon: Dictionary for convergence criteria.
        - Keys:
            - acc_len:  minimum number of accepted steps
            - prev_len: last n (accepted) samples to examine for convergence
            - f_sd: standard deviation of prev_len
    """
    epsilon = {"acc_len": 100, "prev_len": 50, "f_sd": f_sd}
    random.seed(0)
    if init_ca:
        Tract.createAllTracts()
        CommunityArea.createAllCAs(Tract.tracts)
    ##singleFeatureForStudy = CommunityArea.singleFeature
    # targetName = 'total' # train_average_house_price
    M = 500
    T = Tt
    lmbda = lmbd
    CA_maxsize = 30
    # Plot original community population distribution
    CommunityArea.visualizePopDist(iter_cnt=0,fname=project_name+'-orig-pop-distribution')
    CA_maxsize = 30
    mae1, _, _, errors1 = Linear_regression_training(CommunityArea.features, targetName)
    pop_sd_1 = np.std(CommunityArea.features['population'])
    cnt = 0
    iter_cnt = 0
    mae_series = [mae1]
    sd_series = [pop_sd_1]
    mae_index = [0]
    f_series = [get_f(ae=mae1, T=T, penalty=pop_sd_1, log=True, lmbda=lmbda)]


def get_f(ae, T, penalty=None, log=True, lmbda=0.75):
    """
    compute the "energy function F".

    :param ae: Error measurement
    :param T: Temperature parameter
    :param penalty: value to penalize constrain object
    :param log: (Bool) Return f on log scale
    :param lmbda: regularization on penalty term
    :return: the 'energy' of a given state
    """
    if penalty is None:
        penalty = 0

    if log:
        return -(ae + lmbda * penalty) / T
    else:
        return np.exp(-(ae + lmbda * penalty) / T)


def get_gamma(f_current, f_proposed, symmetric=True, log=True, q_proposed_given_current=None,
              q_current_given_proposed=None):
    """
    Compute gamma to be used in acceptance probability of proposed state
    :param f_current: f ("energy function") of current state
    :param f_proposed: f ("energy function") of proposed state
    :param log: (bool) Are f_current and f_proposed on log scale?
    :return: value for gamma, or probability of accepting proposed state
    """
    if symmetric:
        if log:
            alpha = f_proposed - f_current
            gamma = np.min((0, alpha))
        else:
            alpha = f_proposed / f_current
            gamma = np.min((1, alpha))
    else:
        if log:
            alpha = f_proposed - f_current + q_current_given_proposed - q_proposed_given_current
            gamma = np.min((0, alpha))
        else:
            alpha = (f_proposed * q_current_given_proposed) / (f_current * q_proposed_given_current)
            gamma = np.min((1, alpha))

    return gamma


def softmax(x, log=False):
    """
    Compute softmax of a vector x. Always subtract out max(x) to make more numerically stable.
    :param x: (array-like) vector x
    :param log: (bool) return log of softmax function
    :return: Vector of probabilities using softmax function
    """

    # Numerically stable softmax: softmax_stable = softmax(x - max_i(x))
    max_x = np.max(x)
    x_centered = (x - max_x).astype('float')
    if log:
        log_sum = np.log(np.sum(np.exp(x_centered)))
        return x_centered - log_sum
    else:
        exp_X = np.exp(x_centered)
        return exp_X / np.sum(exp_X)


def isConvergent(epsilon, series, iter=None, max_iteration=None):
    if iter and max_iteration:
        if iter > max_iteration:
            return True

    if len(series) > epsilon['acc_len']:
        stdQs = np.std(series[-epsilon["prev_len"]:])
        avgQs = np.abs(np.mean(series[-epsilon["prev_len"]:]))
        ratio = stdQs / avgQs
        if ratio < epsilon["f_sd"]:
            print("Converged with std {}, abs avg {}, std/avg {}".format(stdQs, avgQs, ratio))
            return True
    return False


def writeBetasToFile(project_name, betas):
    fname = 'output/' + project_name + "-regression_coef.csv"
    betas.to_csv(fname)


def softmaxSamplingScheme(errors, community_structure_dict, boundary_tracts, query_ca_prob=None, log=True):
    """
    Function to hierarchicaly sample (1) Communities, (2) tracts within the given community
    :param errors: Vector of errors for softmax
    :param community_structure_dict: Dictionary of community objects -- reflective of conditional states community
    structure. e.g., x'|x  or x|x'
    :param boundary_tracts: List of tracts on boundary, given existing community structure
    :param query_ca_prob:
                    If None: return the probability of randomly selected
                    else: return probability of selecting given tract ID

    :return:
        t: randomly sampled tract (within sampled community
        sample_ca_id: The selected community ID number. If query_ca_prob == None, this is randomly selected. Else return
                query_ca_prob
        sample_ca_prob: Likelihood that sampled community area, i, was sampled:
                i.e., p(CA_i)
        tract_prob: Likelihood (uniform) that tract, j, was sampled conditinal on the sampled community:
                i.e., p(t_j | CA_i)

    """

    # Compute softmax of errors for sampling probabilities
    ca_probs = softmax(errors, log=log)

    if log:
        ca_choice_probs = np.exp(ca_probs)
    else:
        ca_choice_probs = ca_probs

    # Sample community -- probabilities derived from softmax of regression errors
    if query_ca_prob is not None:
        sample_ca_id = query_ca_prob
    else:
        sample_ca_id = np.random.choice(a=list(community_structure_dict.keys()), size=1, replace=False, p=ca_choice_probs)[0]

    # Collect tracts within sampled community area that lie on community boundary
    sample_ca_boundary_tracts = []

    for tract in boundary_tracts:
        if tract.CA == sample_ca_id:
            sample_ca_boundary_tracts.append(tract)

    # Sample tract (on boundary) within previously sampled community area
    t = np.random.choice(a=sample_ca_boundary_tracts, size=1, replace=False)[0]
    try:
        sample_ca_prob = ca_probs.loc[sample_ca_id]
    except KeyError:
        stop = 0

    tract_prob = 1 / float(len(sample_ca_boundary_tracts))
    if log:
        tract_prob = np.log(tract_prob)

    return t, sample_ca_id, sample_ca_prob, tract_prob


def mcmcSamplerUniform(sample_func, update_sample_weight_func, project_name, targetName):
    """=
    MCMC search for optimal solution.
    Input:
        sample_func is the sample proposal method.
        update_sample_weight_func updates sampling auxilary variables.
        targetName: Predicted Value. y in regression model

    Output:
        Optimal partition plot and training error decreasing trend.
    """
    global mae1, cnt, iter_cnt, pop_sd_1
    print("# sampling")
    while cnt <= M:
        cnt += 1
        iter_cnt += 1
        # sample a boundary tract
        t = sample_func(Tract.boundarySet, 1)[0]
        t_flip_candidate = set()
        for n in t.neighbors:
            if n.CA != t.CA and n.CA not in t_flip_candidate:
                t_flip_candidate.add(n.CA)
        # sample a CA assignment to flip
        new_caid = t_flip_candidate.pop() if len(t_flip_candidate) == 1 else random.sample(t_flip_candidate, 1)[0]
        prv_caid = t.CA
        # check whether spatial continuity is guaranteed, if t is flipped
        ca_tocheck = CommunityArea.CAs[prv_caid].tracts
        del ca_tocheck[t.id]
        resulted_shape = unary_union([e.polygon for e in ca_tocheck.values()])
        ca_tocheck[t.id] = t
        if resulted_shape.geom_type == 'MultiPolygon':
            continue
        # CA size constraint
        if len(CommunityArea.CAs[new_caid].tracts) > CA_maxsize \
                or len(CommunityArea.CAs[prv_caid].tracts) <= 1:
            continue

        # update communities features for evaluation
        t.CA = new_caid
        CommunityArea.updateCAFeatures(t, prv_caid, new_caid)
        # Get updated variance of population distribution
        pop_sd_2 = np.std(CommunityArea.features['population'])
        # evaluate new partition
        mae2, _, _, _ = Linear_regression_training(CommunityArea.features, targetName)
        # Calculate acceptance probability --> Put on log scale
        # calculate f ('energy') of current and proposed states
        f_current = get_f(ae=mae1, T=T, penalty=pop_sd_1, log=True, lmbda=lmbda)
        f_proposed = get_f(ae=mae2, T=T, penalty= pop_sd_2, log=True, lmbda=lmbda)

        # Compute gamma for acceptance probability
        #
        gamma = get_gamma(f_current=f_current, f_proposed=f_proposed, log=True)
        # Generate random number on log scale
        sr = np.log(random.random())
        update_sample_weight_func(mae1, mae2, t)

        if sr < gamma:  # made progress
            mae_series.append(mae2)
            sd_series.append(pop_sd_2)
            f_series.append(f_proposed)
            # Update error, variance
            mae1, pop_sd_1 = mae2, pop_sd_2
            mae_index.append(iter_cnt)

            # update tract boundary set for next round sampling
            Tract.updateBoundarySet(t)
            cnt = 0  # reset counter
            if iter_cnt % 50 == 0:
                print("Iteration {}: {} --> {}".format(iter_cnt, mae1, mae2))

            if isConvergent(epsilon, f_series):
                # when mae converges
                msg = "converge in {} samples with {} acceptances sample conversion rate {}"\
                    .format(iter_cnt, len(mae_series), len(mae_series) / float(iter_cnt))
                print(msg)
                logging.info(msg)
                logging.info('Population standard deviation: %s', pop_sd_1)
                CommunityArea.visualizeCAs(iter_cnt=None, fname=project_name + "-CAs-iter-final.png")
                CommunityArea.visualizePopDist(iter_cnt=None, fname=project_name + '-pop-distribution-final')
                break

        else:
            # restore communities features
            t.CA = prv_caid
            CommunityArea.updateCAFeatures(t, new_caid, prv_caid)

        if iter_cnt % 500 == 0:
            CommunityArea.visualizeCAs(iter_cnt=iter_cnt, fname=project_name + "-CAs-iter-progress.png")
            CommunityArea.visualizePopDist(iter_cnt=iter_cnt, fname=project_name + '-pop-distribution-iter-progress')
            plotMcmcDiagnostics(iter_cnt=iter_cnt, mae_index=mae_index, error_array=mae_series, std_array=sd_series,
                                f_array=f_series, lmbda=lmbda, fname=project_name + '-mcmc-diagnostics-progess-{}'.format(iter_cnt))

    if cnt > M:
        print("Cannot find better flip within {} steps".format(cnt))


def mcmcSamplerSoftmax(project_name, targetName):
    """
    MCMC search for optimal solution.
    Input:
        project_name: prefix for output files
        targetName: Predicted value. y in regression model
    Output:
        Optimal partition plot and training error decreasing trend.
    """
    global mae1, errors1, cnt, iter_cnt, pop_sd_1
    print("# sampling")
    while cnt <= M:
        cnt += 1
        iter_cnt += 1

        ## Hierearchicaly sample community, then boundary tract within community
        t, sample_ca_id, log_sample_ca_prob, log_tract_prob = softmaxSamplingScheme(errors=errors1,
                                                                                    community_structure_dict=CommunityArea.CAs,
                                                                                    boundary_tracts=Tract.boundarySet,
                                                                                    log=True)

        # Find neighbors that lie in different community
        t_flip_candidate = set()
        for n in t.neighbors:
            if n.CA != t.CA and n.CA not in t_flip_candidate:
                t_flip_candidate.add(n.CA)
        # sample a CA assignment to flip
        new_caid = t_flip_candidate.pop() if len(t_flip_candidate) == 1 else random.sample(t_flip_candidate, 1)[0]
        prv_caid = t.CA
        # check wether spatial continuity is guaranteed, if t is flipped
        ca_tocheck = CommunityArea.CAs[prv_caid].tracts
        del ca_tocheck[t.id]
        resulted_shape = unary_union([e.polygon for e in ca_tocheck.values()])
        ca_tocheck[t.id] = t
        if resulted_shape.geom_type == 'MultiPolygon':
            continue
        # CA size constraint
        if len(CommunityArea.CAs[new_caid].tracts) > CA_maxsize \
                or len(CommunityArea.CAs[prv_caid].tracts) <= 1:
            continue

        # Update current state to proposed state
        t.CA = new_caid

        CommunityArea.updateCAFeatures(t, prv_caid, new_caid)
        # update tract boundary set for next round sampling
        Tract.updateBoundarySet(t)

        # Get updated variance of population distribution
        pop_sd_2 = np.std(CommunityArea.features['population'])
        # evaluate new partition
        mae2, _, _, errors2 = Linear_regression_training(CommunityArea.features, targetName)
        # Calculate acceptance probability --> Put on log scale
        # calculate f ('energy') of current and proposed states
        f_current = get_f(ae=mae1, T=T, penalty=pop_sd_1, log=True, lmbda=lmbda)
        f_proposed = get_f(ae=mae2, T=T, penalty=pop_sd_2, log=True, lmbda=lmbda)

        # We need to compute Q to get gamma, since Q is non-symmetric under the softmax sampling scheme
        log_q_proposed_given_current = log_sample_ca_prob + log_tract_prob

        # Reverse conditioning to get q(z | z'); i.e., probability of current state given proposed state
        if prv_caid in errors2.index:
            _, _, log_sample_ca_prob_reverse, log_tract_prob_reverse = softmaxSamplingScheme(errors=errors2,
                                                                                             community_structure_dict=CommunityArea.CAs,
                                                                                             boundary_tracts=Tract.boundarySet,
                                                                                             query_ca_prob=prv_caid,
                                                                                             log=True)
        else:
            log_sample_ca_prob_reverse = -np.inf
            log_tract_prob_reverse = -np.inf

        log_q_current_given_proposed = log_sample_ca_prob_reverse + log_tract_prob_reverse

        # Compute gamma for acceptance probability
        gamma = get_gamma(f_current=f_current,
                          f_proposed=f_proposed,
                          log=True,
                          symmetric=False,
                          q_current_given_proposed=log_q_current_given_proposed,
                          q_proposed_given_current=log_q_proposed_given_current)

        # Generate random number on log scale
        sr = np.log(random.random())

        if sr < gamma:  # made progress
            mae_series.append(mae2)
            sd_series.append(pop_sd_2)
            f_series.append(f_proposed)
            # writeBetasToFile(project_name, regression_coeff)
            # Update error, variance
            mae1, pop_sd_1, errors1 = mae2, pop_sd_2, errors2
            mae_index.append(iter_cnt)

            cnt = 0  # reset counter
            if iter_cnt % 50 == 0:
                print("Iteration {}: {} --> {}".format(iter_cnt, mae1, mae2))

            if isConvergent(epsilon, f_series):
                # when mae converges
                print("converge in {} samples with {} acceptances \
                    sample conversion rate {}".format(iter_cnt, len(mae_series),
                                                      len(mae_series) / float(iter_cnt)))
                CommunityArea.visualizeCAs(iter_cnt=None, fname=project_name + "-CAs-iter-final.png")
                CommunityArea.visualizePopDist(iter_cnt=None, fname=project_name + '-pop-distribution-final')
                break

        else:
            # restore community-tract structure to original state
            t.CA = prv_caid
            CommunityArea.updateCAFeatures(t, new_caid, prv_caid)
            Tract.updateBoundarySet(t)

        if iter_cnt % 500 == 0:
            CommunityArea.visualizeCAs(iter_cnt=iter_cnt,
                                       fname=project_name + "-CAs-iter-progress.png")
            CommunityArea.visualizePopDist(iter_cnt=iter_cnt,
                                           fname=project_name + '-pop-distribution-iter-progress')
            plotMcmcDiagnostics(iter_cnt=iter_cnt,
                                mae_index=mae_index,
                                error_array=mae_series,
                                std_array=sd_series,
                                f_array=f_series,
                                lmbda=lmbda,
                                fname=project_name + '-mcmc-diagnostics-progess-{}'.format(iter_cnt))
    if cnt > M:
        print("Cannot find better flip within {} steps".format(cnt))


def leaveOneOut_evaluation(targetName, info_str="optimal boundary"):
    """
    Leave-one-out evaluation the current partition with next year crime rate.
    """
    if targetName == 'test_crime':
        # If doing crime task, re-load task at given year
        CommunityArea._initializeCAfeatures()

    print("leave one out with {} in {}".format(info_str, targetName))
    reg_eval = Linear_regression_evaluation(CommunityArea.features, targetName)
    print(reg_eval)
    return reg_eval


def naive_MCMC(project_name, targetName='train_crime', lmbda=0.75, f_sd=1.5, Tt=10, init_ca=True):
    """
    Run naive MCMC
    :param project_name: string
    :param targetName: Predicted value. y in regression model
    :return:
    """
    if targetName not in ['train_crime', 'train_average_house_price']:
        raise Exception("targetName must be train_crime (for crime) or train_average_house_price (for house price)")

    initialize(project_name, targetName, lmbda, f_sd, Tt, init_ca)
    mcmcSamplerUniform(random.sample, lambda ae1, ae2, t: 1, project_name=project_name, targetName=targetName)
    rmse, mae, mape, r2, residuals = leaveOneOut_evaluation(targetName=targetName.replace('train', 'test'))
    plotMcmcDiagnostics(iter_cnt=None,
                        mae_index=mae_index,
                        error_array=mae_series,
                        std_array=sd_series,
                        f_array=f_series,
                        lmbda=lmbda,
                        fname=project_name + "-mcmc-diagnostics-final")
    writeSimulationOutput(project_name=project_name,
                          mae=mae, rmse=rmse, mape=mape, residual=residuals,
                          n_iter_conv=iter_cnt,
                          accept_rate=len(mae_series) / float(iter_cnt))
    Tract.writePartition(fname=project_name + "-final-partition.txt")
    return rmse, mae, mape, r2, residuals


def adaptive_MCMC():
    initialize()
    # initialize adapative sampling variable
    ntrct = len(Tract.tracts)
    tractWeights = dict(zip(Tract.tracts.keys(), [1.0] * ntrct))

    def adaptive_sample(tractSet, k):
        tractIDs = [t.id for t in tractSet]
        sampleWeights = [tractWeights[tid] for tid in tractIDs]
        tmp = random.uniform(0, sum(sampleWeights))

        for tid in tractIDs:
            if tmp < tractWeights[tid]:
                return [Tract.tracts[tid]]
            else:
                tmp -= tractWeights[tid]
        return [None]

    def update_tractWeight(ae1, ae2, t):
        if ae1 < ae2:
            tractWeights[t.id] *= 0.8
        else:
            tractWeights[t.id] *= 1 / 0.8

    mcmcSamplerUniform(adaptive_sample, update_tractWeight)
    plotMcmcDiagnostics(mae_index=mae_index, error_array=mae_series, std_array=sd_series)


def MCMC_softmax_proposal(project_name, targetName='train_crime', lmbda=0.75, f_sd=1.5, Tt=10, init_ca=True):
    """
    Run guided MCMC
    :param project_name: string
    :param targetName: Predicted value. y in regression model
    :return:
    """
    global mae_series, sd_series, f_series
    if targetName not in ['train_crime', 'train_average_house_price']:
        raise Exception("targetName must be train_crime (for crime) or train_average_house_price (for house price)")

    initialize(project_name, targetName, lmbda, f_sd, Tt, init_ca)
    mcmcSamplerSoftmax(project_name, targetName=targetName)
    rmse, mae, mape, r2, residuals = leaveOneOut_evaluation(targetName.replace('train', 'test'))

    plotMcmcDiagnostics(iter_cnt=None,
                        mae_index=mae_index,
                        error_array=mae_series,
                        f_array=f_series,
                        std_array=sd_series,
                        lmbda=lmbda,
                        fname=project_name + "-mcmc-diagnostics-final")
    writeSimulationOutput(project_name=project_name,
                          mae=mae, rmse=rmse, mape=mape, residual=residuals,
                          n_iter_conv=iter_cnt,
                          accept_rate=len(mae_series) / float(iter_cnt))

    Tract.writePartition(fname=project_name + "-final-partition.txt")
    return rmse, mae, mape, r2, residuals


if __name__ == '__main__':
    import random
    random.seed(100)
    results = np.zeros(shape=(10, 1, 5))
    for i in range(1, 11):
        version = "v{}".format(i)
        print("-----{}-----".format(version))

        # Crime
        # crime_naive = naive_MCMC('crime-naive-{}'.format(version), targetName='train_crime', lmbda=0.03, f_sd=0.008, Tt=1)
        crime_soft = MCMC_softmax_proposal('crime-softmax-{}'.format(version), targetName='train_crime', lmbda=0.03, f_sd=0.008, Tt=1)

        # House Prices
        # price_naive = naive_MCMC('house-price-naive-{}'.format(version), targetName='train_average_house_price', lmbda=0.0004, f_sd=0.008, Tt=1)
        # price_soft = MCMC_softmax_proposal('house-price-softmax-{}'.format(version), targetName='train_average_house_price', lmbda=0.0004, f_sd=0.008, Tt=1)


        # results[i-1, 0, :] = crime_naive
        results[i-1, 0, :] = crime_soft
        # results[i-1, 2, :] = price_naive
        # results[i-1, 3, :] = price_soft

    means = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    print(means)
    print(std)
