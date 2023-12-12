import csv
from tract import Tract
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
from util.feature_util import *


class CommunityArea:

    def __init__(self, caID):
        self.id = caID
        self.tracts = {}

    def addTract(self, tID, trct):
        self.tracts[tID] = trct

    def initializeField(self):
        """
        Prerequisite:
            all tracts are added to corresponding CA.
        Goal:
            initialize the boundary and features of each CA.
        """
        self.polygon = cascaded_union([e.polygon for e in self.tracts.values()])
        tract_features = CommunityArea.features_raw.loc[self.tracts.keys()]

        feat_series = tract_features.sum(axis=0)
        feat_vals = feat_series.to_numpy()[None]
        fdf = pd.DataFrame(feat_vals, columns=feat_series.index.to_numpy(), index=[self.id])
        # calculate the average house price (training and testing)
        fdf['train_average_house_price'] = fdf['train_price'] / fdf['train_count']
        fdf['train_average_house_price'] = \
            fdf['train_average_house_price'].replace([np.inf, -np.inf, np.nan], CommunityArea.default_house_price_train)
        fdf['test_average_house_price'] = fdf['test_price'] / fdf['test_count']
        fdf['test_average_house_price'] = \
            fdf['test_average_house_price'].replace([np.inf, -np.inf, np.nan], CommunityArea.default_house_price_test)
        self.features = fdf


    @classmethod
    def createAllCAs(cls, tracts):
        """
        tracts:
            a dict of Tract, each of which has CA assignment.
        Output:
            a dict of CAs
        """
        CAs = {}
        # initialize boundary
        for tID, trct in tracts.items():
            assert trct.CA != None
            if trct.CA not in CAs:
                ca = CommunityArea(trct.CA)
                ca.addTract(tID, trct)
                CAs[trct.CA] = ca
            else:
                CAs[trct.CA].addTract(tID, trct)
        cls.CAs = CAs
        cls._initializeCAfeatures()
        return CAs

    @classmethod
    def _initializeCAfeatures(cls):
        cls.features_raw = Tract.features if hasattr(Tract, "features") else Tract.generateFeatures()
        cls.default_house_price_train = np.sum(cls.features_raw['train_price']) / np.sum(cls.features_raw['train_count'])
        cls.default_house_price_test = np.sum(cls.features_raw['test_price']) / np.sum(cls.features_raw['test_count'])
        cls.features_ca_dict = {}
        for ca in cls.CAs.values():
            ca.initializeField()
            cls.features_ca_dict[ca.id] = ca.features
        cls.features = pd.concat(cls.features_ca_dict.values())


    @classmethod
    def updateCAFeatures(cls, tract, prv_CAid, new_CAid):
        """
        Update the CA features, when one tract is flipped from prv_CA to new_CA.
        """
        prv_CA = cls.CAs[prv_CAid]
        del prv_CA.tracts[tract.id]
        prv_CA.initializeField()
        cls.features_ca_dict[prv_CAid] = prv_CA.features

        new_CA = cls.CAs[new_CAid]
        new_CA.tracts[tract.id] = tract
        new_CA.initializeField()
        cls.features_ca_dict[new_CAid] = new_CA.features

        # convert dict of pandas.Series into DataFrame
        cls.features = pd.concat(cls.features_ca_dict.values())

    @classmethod
    def rand_init_communities(cls, target_m):
        all_ca = cls.CAs
        features_ca_dict = cls.features_ca_dict
        M = len(all_ca)
        del_ca_list = list()

        while M > target_m:
            # select random community
            ca_rand_id = np.random.permutation(all_ca.keys())[0]
            ca_rand = all_ca[ca_rand_id]

            # get all adjacent communities to selected community
            adj_ca = set()
            for tract_id, tract in ca_rand.tracts.items():
                for neighbor in tract.neighbors:
                    if neighbor.CA != ca_rand_id:
                        adj_ca.add(neighbor.CA)

            # select random adjacent community
            adj_ca = list(adj_ca)
            new_ca = np.random.permutation(adj_ca)[0]
            print("Merging community #{} into #{}".format(ca_rand_id, new_ca))
            # update each tract in selected community area
            for tract_id, tract in ca_rand.tracts.items():
                tract.CA = new_ca
                cls.updateCAFeatures(tract, prv_CAid=ca_rand_id, new_CAid=new_ca)
                Tract.updateBoundarySet(tract)

            # remove selected commununity
            del all_ca[ca_rand_id]
            del features_ca_dict[ca_rand_id]
            del_ca_list.append(ca_rand_id)

            M -= 1

        cls.CAs = all_ca
        cls.features_ca_dict = features_ca_dict
        cls.features = cls.features.drop(del_ca_list, axis=0)

    @classmethod
    def get_ca_tract_dict(cls):
        ca_to_tract_map = dict()

        for ca_id, ca in cls.CAs.items():
            tract_list = list()
            for t_id, tract in ca.tracts.items():
                tract_list.append(t_id)

            ca_to_tract_map[ca_id] = tract_list

        return ca_to_tract_map

    @classmethod
    def visualizeCAs(cls,
                     iter_cnt=None,
                     CAs=None,
                     fname="CAs.png",
                     plot_measure=None,
                     labels=False,
                     title=None,
                     case_study=False,
                     comm_to_plot=None,
                     jitter_labels=False,
                     marker=None):
        """
        Class method to plot community boundaries. Arguments available for plotting a continuous measure over the
        community areas
        :param iter_cnt: (int) Specifies the current iteration
                            Use if plotting updates in a sequential algorithm; i.e., MCMC.
                            will be inserted into plot title if title = None
        :param CAs: (dict) Dictionary of communities. If none is given, default is class attribute CAs
        :param fname: (str) name of .png file to be written to disk
        :param plot_measure: (Series) Use if we want to visually scale a continous measure by community are (i.e., heat map)
                             plot_measure is pd.Series instance with the community ID's as the index, and the measure as the
                             values
        :param labels: (bool) Used in conjuction with plot_measure. If True, plot the community ID and measure value at the
                                centroid of each community area
        :param title: (str) Title for plot saved .png. If title = '', then no title, If title is None, then use default
                            title that prints iteration count
         :param marker: (tuple) GPS coordinates to mark on map w/ star
        :return:
        """
        if CAs == None:
            CAs = cls.CAs
        if iter_cnt is None:
            iter_cnt = "completed"

        from descartes import PolygonPatch
        f = plt.figure(figsize=(16, 16))
        ax = f.gca()

        if plot_measure is not None:
            # Create heat map
            # Scale colors by sorted values of plot_measure
            col_gradient = np.linspace(0.05, 1, len(plot_measure))
            plot_measure.sort_values(ascending=False, inplace=True)
            # Create color map (dict). Keys: Community ID - Values: Green values (in RGB) in [0,1] interval
            color_map = dict(zip(plot_measure.index, col_gradient))

            if case_study:
                for k, t in CAs.items():
                    if t.id in comm_to_plot:
                        lw = .75
                        ca_id = t.id
                        ax.add_patch(PolygonPatch(t.polygon, alpha=0.85,
                                                  fc=(1, color_map[k], 0),
                                                  edgecolor='black',
                                                  linewidth=lw))
                        if labels:
                            if jitter_labels:
                                x = t.polygon.centroid.x + np.random.uniform(0, .01)
                                y = t.polygon.centroid.y + np.random.uniform(0, .01)
                            else:
                                x = t.polygon.centroid.x
                                y = t.polygon.centroid.y

                            ax.text(x, y,
                                    ca_id,
                                    horizontalalignment='center',
                                    verticalalignment='center', fontsize=48)

                    else:
                        pass

                if marker is not None:
                    plt.plot(marker[0], marker[1], marker='*', color='green', ms=40)

            else:
                for k, t in CAs.items():
                    ax.add_patch(PolygonPatch(t.polygon, alpha=0.85,
                                              fc=(1, color_map[k], 0),
                                              edgecolor='black',
                                              linewidth=.5))

                    if labels:
                        ax.text(t.polygon.centroid.x, t.polygon.centroid.y,
                                t.id,
                                horizontalalignment='center',
                                verticalalignment='center', fontsize=18)

        else:

            for k, t in CAs.items():
                ax.add_patch(PolygonPatch(t.polygon, alpha=1, fc='#E9FFBE', lw=1.5, edgecolor='black'))

                if labels:
                    # Label plot with community ids at each respective centroid
                    ax.text(t.polygon.centroid.x,
                            t.polygon.centroid.y,
                            int(t.id),
                            horizontalalignment='center',
                            verticalalignment='center', fontsize=12)

        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        if title is None:
            plt.title('Community Structure -- Iterations: {}'.format(iter_cnt))
        elif title == '':
            pass
        else:
            plt.title(title, fontsize=20)
        plt.tight_layout()
        # plt.show()
        plt.savefig("plots/" + fname)
        plt.close()
        plt.clf()

    @classmethod
    def visualizePopDist(cls, fname, iter_cnt=None):
        if iter_cnt is None:
            iter_cnt = "completed"
        pop_df = pd.DataFrame(cls.features['population'])
        pop_df.plot(kind='barh', figsize=(16, 12))
        plt.title('Population Distribution -- Iterations: {}'.format(iter_cnt))
        plt.savefig("plots/" + fname + ".png")
        plt.close()
        plt.clf()

def fig_clustering_baseline2(task, n_sim, cluster_X=True, cluster_y=False):
    """
    A second function to compute baselines using classical clustering techniques (k-means,agglomerative,spectral)
    This function differs from fig_clustering_baseline() in that we can now specify on which dimensions we wish to cluster.
    More specifically, we can cluster only X (the feature matrix) only, y (the target variable) only, or on both X AND y.

    ** Note: if both cluster_X and cluster_y are True, then cluster on both X and y..

    :param task: (str) must be either 'crime', or 'house-price' - designates the prediction  task
    :param cluster_X: (bool) Boolean to cluster on X
    :param cluster_y: (bool) Boolean to cluster on y
    :param cluster_y: (bool) Whether or not to initialize city structure from within function:
                        - if False, assumes that some Tract and CommunityArea classses are
                        initialized before executing function
    :return: None
    """

    from regression import Linear_regression_evaluation

    Tract.createAllTracts()
    Tract.generateFeatures()

    tract_task_y_map = {'house-price': 'test_price', 'crime': 'test_crime'}
    ca_task_y_map = {'house-price': 'test_average_house_price', 'crime': 'test_crime'}
    y_tract = tract_task_y_map[task]
    y_ca = ca_task_y_map[task]

    results = np.zeros(shape=(n_sim, 4, 5))

    if cluster_X == True and cluster_y == False:
        tag = 'based_X'
    elif cluster_X == False and cluster_y == True:
        tag = 'based_Y'
    else:
        tag = 'based_XY'

    for i in range(n_sim):
        print("------ ITERATION {} -------  ".format(i+1))
        print("--> {} error:".format(task))

        print("-------Admin. Boundary-------")
        CommunityArea.createAllCAs(Tract.tracts)
        # CommunityArea.get_ca_tract_dict()
        # CommunityArea.visualizeCAs(fname="admin_CA_{}_{}.png".format(tag, task), title='')
        admin_reg = Linear_regression_evaluation(CommunityArea.features, targetName=y_ca, taged='admin'+str(i+1))
        print(admin_reg)

        print("-------Kmeans Clustering-------")
        KM_tract_CA_map = Tract.kMeansClustering(cluster_X=cluster_X, cluster_y=cluster_y, y=y_tract)
        writePartitions('./output/baseline_clustering/{}_Kmeans_v{}'.format(task, i+1), KM_tract_CA_map)
        CommunityArea.createAllCAs(Tract.tracts)
        CommunityArea.visualizeCAs(fname="kmeans_CA_{}_{}_v{}.png".format(tag, task, i+1), title='')
        k_means_reg = Linear_regression_evaluation(CommunityArea.features, targetName=y_ca, taged='kmans'+str(i+1))
        print(k_means_reg)

        print("-------Agglomerative Clustering-------")
        AGG_tract_CA_map = Tract.agglomerativeClustering(cluster_X=cluster_X, cluster_y=cluster_y, y=y_tract)
        writePartitions('./output/baseline_clustering/{}_Agg_v{}'.format(task, i + 1), AGG_tract_CA_map)
        CommunityArea.createAllCAs(Tract.tracts)
        agg_reg = Linear_regression_evaluation(CommunityArea.features, targetName=y_ca, taged='agg'+str(i+1))
        CommunityArea.visualizeCAs(fname="agg_CA_{}_{}_v{}.png".format(tag, task, i+1), title='')
        print(agg_reg)

        print("-------Spectral Clustering-------")
        Spectral_tract_CA_map =Tract.spectralClustering(cluster_X=cluster_X, cluster_y=cluster_y, y=y_tract)
        writePartitions('./output/baseline_clustering/{}_Spectral_v{}'.format(task, i + 1), Spectral_tract_CA_map)
        CommunityArea.createAllCAs(Tract.tracts)
        spectral_reg = Linear_regression_evaluation(CommunityArea.features, targetName=y_ca, taged='spectral'+str(i+1))
        CommunityArea.visualizeCAs(fname="spectral_CA_{}_{}__v{}.png".format(tag, task, i+1), title='')
        print(spectral_reg)
        print("\n")

        results[i, 0, :] = admin_reg
        results[i, 1, :] = k_means_reg
        results[i, 2, :] = agg_reg
        results[i, 3, :] = spectral_reg

    print(results)

    means = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    means = pd.DataFrame(means, index=['admin', 'kmeans', 'agglomerative', 'spectral'], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    std = pd.DataFrame(std, index=['admin', 'kmeans', 'agglomerative', 'spectral'], columns=['rmse', 'mae', 'mape', 'r2', 'residuals'])
    return results, means, std

def writePartitions(fname, map_index):
    with open(fname + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in map_index.items():
            writer.writerow(row)



if __name__ == '__main__':
    # task ['house-price', 'crime']
    task = 'crime'
    results, means, std = fig_clustering_baseline2(task=task, n_sim=10, cluster_X=True, cluster_y=True)
    means.to_csv("output/baselines-mean-results-{}-{}.csv".format('based_XY', task))
    std.to_csv("output/baselines-std-results-{}-{}.csv".format('based_XY', task))

    # mark = [[True, False], [False, True], [True, True]]
    # tags = ['based_X', 'based_Y', 'based_XY']
    # for item in mark:
    #     results, means, std = fig_clustering_baseline2(task=task, n_sim=10, cluster_X=item[0], cluster_y=item[1])
    #     means.to_csv("output/baselines-mean-results-{}-{}.csv".format(tags[mark.index(item)],task))
    #     std.to_csv("output/baselines-std-results-{}-{}.csv".format(tags[mark.index(item)], task))
