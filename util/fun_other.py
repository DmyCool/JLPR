import numpy as np
from sklearn.manifold import TSNE
import torch_geometric.utils as utils
import matplotlib.pyplot as plt
import networkx as nx
import torch
import os
import geopandas as gpd
import libpysal as lps
import pandas as pd



def load_data():
    adj_mtx = np.load('../data/adj_mtx.npy')
    flow_mtx = np.load('../data/flow_count_similarity_mtx.npy')

    img = np.load('../data/img_feat_mtx.npy')
    poi = np.load('../data/poi_emb_mtx.npy')
    feat_emb = np.concatenate((poi, img), axis=-1)

    crime_train = np.load('../data/crime_train.npy')
    crime_test = np.load('../data/crime_test.npy')
    y_crime = np.load('../data/crime.npy')
    return adj_mtx, flow_mtx, img, poi, feat_emb, crime_train, crime_test, y_crime



def normalize_features(feat):
    '''row-normalize feature matrix'''
    colvar = torch.var(feat, dim=-1, keepdim=True)
    colmean = torch.mean(feat, dim=-1, keepdim=True)
    c_inv = torch.pow(colvar, -0.5)
    c_inv[torch.isinf(c_inv)] = 0.
    feat = torch.mul((feat-colmean), c_inv)
    return feat

def normalize_mtx(mtx):
    row_sums = torch.sum(mtx, dim=1).reshape(-1, 1)
    mtx_norm = mtx / row_sums
    idx_nan = torch.isnan(mtx_norm)
    mtx_norm[idx_nan] = 0.0
    return mtx_norm



def draw_graph(data):
    networkX_graph = utils.to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)
    # print(networkX_graph)
    plt.figure(figsize=(80, 80), dpi=80)
    nx.draw(networkX_graph, with_labels=True, font_size=10, node_size=80)
    plt.savefig('draw.png')
    plt.show()


def visualize(h, epoch):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    # plt.xticks([])
    # plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, cmap="Set2")
    figname = './output/tsne_emb/' + str(epoch)+'.png'
    plt.savefig(figname, dpi=300)
    # plt.show()


def visualize_shp_adj(shp_path, adj_name):
    gdf = gpd.read_file(shp_path)

    weight_rook = lps.weights.Rook.from_dataframe(gdf)

    ax = gdf.plot(edgecolor='grey', facecolor='w')
    f, ax = weight_rook.plot(gdf, ax=ax, edge_kws=dict(color='r', linestyle=':', linewidth=1),
                             node_kws=dict(marker='o'))
    ax.set_axis_off()

    neighbor = weight_rook.neighbors

    correlation_matrix = pd.DataFrame([k, e] for k, v in neighbor.items() for e in v)

    adj_mtx = pd.crosstab(correlation_matrix[0], correlation_matrix[1])

    adj_mtx.to_csv(adj_name)

    gdf.plot()
    plt.show()




def writePartition(epoch, tract_id, s_label, m_cluster):
    partition_path_dir = './results_record/Partitions/m_' + str(m_cluster)
    if not os.path.exists(partition_path_dir):
        os.mkdir(partition_path_dir)
    fname = 'partition_e_{}.txt'.format(epoch)
    partittion_label = s_label.detach().cpu().numpy()
    tract_to_partition_map = list(zip(tract_id, partittion_label))
    with open(partition_path_dir + fname,'w+') as fp:
        fp.write('\n'.join('%s, %s' % x for x in tract_to_partition_map))

def writeEmbedding(epoch, m_cluster, emb):
    embedding_path_dir = './results_record/Embeddings/m_' + str(m_cluster)
    if not os.path.exists(embedding_path_dir):
        os.mkdir(embedding_path_dir)
    np.save(embedding_path_dir + 'region_emb_e' + str(epoch+1), emb)

def writeMetrics(epoch, m_cluster, mae, rmse, r2, mape):
    metrics_path_dir = './results_record/task_metrics/m_' + str(m_cluster)
    if not os.path.exists(metrics_path_dir):
        os.mkdir(metrics_path_dir)
    f = open(metrics_path_dir + 'task_metrics.txt', 'a+')
    content = str(epoch+1) + ',' + str(mae) + ',' + str(rmse) + ',' + str(r2) + ',' + str(mape) + '\n'
    f.write(content)
    f.close()