import matplotlib
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import shapefile
from util.feature_util import *
from sklearn.cluster import AgglomerativeClustering, KMeans, spectral_clustering
import numpy as np


class Tract:

    def __init__(self, tid, shp, rec=None):
        """Build one tract from the shapefile._Shape object"""
        self.id = tid
        self.bbox = box(*shp.bbox)
        self.polygon = Polygon(shp.points)
        self.centroid = (self.polygon.centroid.x, self.polygon.centroid.y)
        if rec != None:
            self.CA = int(rec[0])
        else:
            self.CA = None
        # for adjacency information
        self.neighbors = []
        self.onEdge = False

    @classmethod
    def get_tract_ca_dict(cls):
        tract_to_ca_map = dict()

        for t_id, tract in cls.tracts.items():
            tract_to_ca_map[t_id] = tract.CA

        return tract_to_ca_map

    @classmethod
    def createAllTracts(cls, fname="./data/Census-Tracts-2010/chicago-tract", calculateAdjacency=True):
        cls.sf = shapefile.Reader(fname)
        tracts = {}
        shps = cls.sf.shapes()
        for idx, shp in enumerate(shps):
            rec = cls.sf.record(idx)
            # tid = int("".join([rec[0], rec[1], rec[2]]))
            tid = int(rec[1])
            trt = Tract(tid, shp, rec)
            tracts[tid] = trt
        cls.tracts = tracts

        cls.tract_index = sorted(cls.tracts.keys())

        if calculateAdjacency:
            cls.spatialAdjacency()
        return tracts

    @classmethod
    def visualizeTracts(cls, tractIDs=None, tractColors=None, fsize=(16,16), fname="tracts.png",labels=False):
        tracts = {}
        if tractIDs == None:
            tracts = cls.tracts
        else:
            for tid in tractIDs:
                tracts[tid] = cls.tracts[tid]
        if tractColors == None:
            tractColors = dict(zip(tracts.keys(), ['#E9FFBE']* len(tracts)))
        print(tractColors)
        from descartes import PolygonPatch
        f = plt.figure(figsize=fsize)
        ax = f.gca()
        for k, t in tracts.items():
            ax.add_patch(PolygonPatch(t.polygon, alpha=1, fc=tractColors[k]))
            if labels:
                ax.text(t.polygon.centroid.x,
                        t.polygon.centroid.y,
                        int(t.id),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=11)
        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(fname)

    @classmethod
    def generateFeatures(cls):
        cls.features = generate_all_features()
        # set tractID as index
        cls.features.index = sorted(cls.tract_index)

        return cls.features

    @classmethod
    def spatialAdjacency(cls):
        """
        Calculate the adjacent tracts.
        """
        for focalKey, focalTract in cls.tracts.items():
            for otherKey, otherTract in cls.tracts.items():
                if otherKey != focalKey and focalTract.polygon.touches(otherTract.polygon):
                    intersec = focalTract.polygon.intersection(otherTract.polygon)
                    if intersec.geom_type != 'Point':
                        focalTract.neighbors.append(otherTract)
        # calculate whether the tract is on CA boundary
        cls.initializeBoundarySet()

    @classmethod
    def initializeBoundarySet(cls):
        """
        Initialize the boundary set on given partitions.
        """
        cls.boundarySet = set()
        for _, t in cls.tracts.items():
            for n in t.neighbors:
                if t.CA != n.CA:
                    t.onEdge = True
                    cls.boundarySet.add(t)
                    break

    @classmethod
    def updateBoundarySet(cls, tract):
        """
        Update bounary set for next round sampling
        """
        tracts_check = [tract] + tract.neighbors
        for t in tracts_check:
            onEdge = False
            for n in t.neighbors:
                if t.CA != n.CA:
                    onEdge = True
                    break
            if not onEdge:
                if t.onEdge:
                    t.onEdge = False
                    cls.boundarySet.remove(t)
            else:
                t.onEdge = True
                cls.boundarySet.add(t)

    @classmethod
    def visualizeTractsAdjacency(cls):
        """
        Plot tract adjacency graph. Each tract is ploted with its centroid.
        The adjacency
        """
        from matplotlib.lines import Line2D
        tracts = cls.tracts
        f = plt.figure(figsize=(16, 16))
        ax = f.gca()
        for _, t in tracts.items():
            for n in t.neighbors:
                ax.add_line(Line2D(*zip(t.centroid, n.centroid)))
        ax.axis('scaled')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("adjacency.png")


    @classmethod
    def getPartition(cls):
        return [cls.tracts[k].CA for k in cls.tract_index]

    @classmethod
    def getTractPosID(cls, t):
        return cls.tract_index.index(t.id)

    @classmethod
    def restorePartition(cls, partition):
        for i, k in enumerate(cls.tract_index):
            cls.tracts[k].CA = partition[i]

    @classmethod
    def writePartition(cls,fname):
        # f = open("output/" + fname,'w')
        tract_ca_assignment = cls.getPartition()
        tract_ca_map = list(zip(cls.tract_index, tract_ca_assignment))
        with open("output/" + fname,'w+') as fp:
            fp.write('\n'.join('%s, %s' % x for x in tract_ca_map))


    @classmethod
    def readPartition(cls,fname):
        f = open("output/" + fname, 'r')
        partition = f.readlines()
        partition_clean = [int(x.rstrip()) for x in partition]
        f.close()
        cls.restorePartition(partition=partition_clean)

    @classmethod
    def agglomerativeClustering(cls, cluster_X=True,cluster_y=False,y=None, algorithm = "ward"):
        '''
        using agglomerative clustering
        :return: tract to CA mapping
        '''
        connectivity, node_value, CA_count, tract_ids = cls.constructConnectivity(Xfeatures=cluster_X,
                                                                                  target_bool=cluster_y,
                                                                                  target=y)


        if algorithm == "ward":
            ward = AgglomerativeClustering(n_clusters=CA_count, linkage="ward",
                                           connectivity=connectivity)
        elif algorithm =="average_cosine":
            ward = AgglomerativeClustering(n_clusters=CA_count, linkage="average",
                                           affinity="cosine",
                                           connectivity=connectivity)
        elif algorithm =="average_cityblock":
            ward = AgglomerativeClustering(n_clusters=CA_count, linkage="average",
                                           affinity="cityblock",
                                           connectivity=connectivity)
        elif algorithm =="complete_cosine":
            ward = AgglomerativeClustering(n_clusters=CA_count, linkage="complete",
                                           affinity="cosine",
                                           connectivity=connectivity)
        else:
            print('ERROR: agglomerative clustering params wrong!')
        ward.fit(node_value)
        labels = ward.labels_
        tract_to_CA_dict = dict(zip(tract_ids,labels))
        cls.updateCA(tract_to_CA_dict)
        return tract_to_CA_dict

    @classmethod
    def kMeansClustering(cls,cluster_X=True,cluster_y=False,y=None):
        """
        cluster tracts using kmeans clustering
        :return:
        """
        connectivity, node_value, CA_count, tract_ids = cls.constructConnectivity(Xfeatures=cluster_X,
                                                                                  target_bool=cluster_y,
                                                                                  target=y)

        km = KMeans(n_clusters=CA_count,init='k-means++')
        km.fit(node_value)
        labels = km.labels_
        tract_to_CA_dict = dict(zip(tract_ids, labels))
        cls.updateCA(tract_to_CA_dict)
        return tract_to_CA_dict

    @classmethod
    def spectralClustering(cls,cluster_X=True,cluster_y=False,y=None,assign_labels='discretize'):
        """
        cluster tracts using spectral clustering
        :return:
        """

        connectivity, node_value, CA_count, tract_ids = cls.constructConnectivity(Xfeatures=cluster_X,
                                                                                  target_bool=cluster_y,
                                                                                  target=y)

        labels = spectral_clustering(connectivity, n_clusters=CA_count,
                                     assign_labels=assign_labels, random_state=None)
        tract_to_CA_dict = dict(zip(tract_ids, labels))
        cls.updateCA(tract_to_CA_dict)
        return tract_to_CA_dict

    @classmethod
    def constructConnectivity(cls,Xfeatures=True,target_bool=False,target=None):
        '''
        Construct connectivity matrix for clustering methods
        :return: Adjacency matrix, node value matrix, number of CA_ids,
                 tract_ids order in cls.items
        '''
        from scipy import sparse
        N = len(cls.tracts)
        I = []
        J = []
        V = []
        X = []
        CA_ids = []
        tract_ids = []

        node_features = list()

        if Xfeatures:
            node_features += list(cls.features.keys()[-3:-1])
        if target_bool and target is not None:
            node_features += [target]

        #TODO: Delete line
        feature_names_all = list(cls.features.columns)

        target_in_names = target in feature_names_all

        for focalKey, focalTract in cls.tracts.items():
            tract_ids.append(focalKey)
            CA_ids.append(focalTract.CA)
        for focalKey, focalTract in cls.tracts.items():
            #X.append(cls.features.loc[focalKey, cls.income_description.keys()])
            X.append(cls.features.loc[focalKey, node_features])
            for neighbor in focalTract.neighbors:
                I.append(tract_ids.index(focalKey))
                J.append(tract_ids.index(neighbor.id))
                V.append(1)

        if Xfeatures == True and target_bool == False:  # cluster based on X
            X = np.array(X)
            X = np.hstack((np.array(list(X[:, 0])), np.array(list(X[:, 1]))))
        elif Xfeatures == False and target_bool == True:  # cluster based on Y
            X = np.array(X)
        else: # cluster based on XY
            X = np.array(X)
            X = np.hstack((np.array(list(X[:, 0])), np.array(list(X[:, 1])), np.expand_dims(np.array(list(X[:, 2])), axis=1)))


        return sparse.coo_matrix((np.array(V), (np.array(I), np.array(J))), shape=(N, N)), \
               X, np.unique(np.array(CA_ids)).size ,tract_ids

    @classmethod
    def updateCA(cls,tract_to_CA_dict):
        '''
        Update the CA id according to tract_CA mapping
        :param tract_to_CA_dict:
        :return:
        '''
        for focalKey, focalTract in cls.tracts.items():
            focalTract.CA = tract_to_CA_dict[focalKey]




if __name__ == '__main__':
    trts0 = Tract.createAllTracts()  # {17031842400: <__main__.Tract object at 0x000001D8CE667708>, 17031840300: <__main__.Tract object at 0x000001D8CE675D08>,...}
    print(trts0)
    Tract.visualizeTracts(tractIDs=trts0)
    Tract.visualizeTractsAdjacency()

    # trts0_features = Tract.generateFeatures()
    # print(trts0_features)
    #tract_to_CA_dict = Tract.agglomerativeClustering()
    # Tract.updateCA(tract_to_CA_dict)
    # trts1 = Tract.tracts


    #tract_to_ca_dict = Tract.kMeansClustering()
    # tract_to_ca_dict = Tract.spectralClustering()