import geopandas as gpd
from gerrychain import Graph
from sklearn.metrics.pairwise import cosine_similarity
from tools import *


def create_tract():
    chicago = gpd.read_file('../data/Census-Tracts-2010/chicago-tract.shp')
    tractid_list = list(set(chicago['geoid10']))  # 801
    tractid_number = [int(i) for i in tractid_list]
    tractid_numbers = sorted(tractid_number)
    return chicago, tractid_numbers


def adj_mtx():
    chicago, _ = create_tract()
    chicago = chicago.sort_values(by='geoid10').reset_index(drop=True)
    graph = Graph.from_geodataframe(chicago)
    adj_mtx = nx.to_numpy_array(graph)  # 获取邻接矩阵
    index_list = chicago.index[(chicago['geoid10'] == '17031760802') | (chicago['geoid10'] == '17031980000')].tolist()
    adj_mtx[index_list[0],index_list[1]] = 1
    adj_mtx[index_list[1],index_list[0]] = 1

    if not os.path.exists('../data/adj_mtx.npy'):
        np.save('../data/adj_mtx.npy', adj_mtx)
    return adj_mtx



def similarity_flow():
    flow_mtx = np.load('../data/flow_mtx.npy')
    node_weight = []  # (startflow, endflow)
    for i in range(flow_mtx.shape[0]):
        startflow = sum(flow_mtx[i, :])
        endflow = sum(flow_mtx[:, i])
        node_weight.append((startflow, endflow))
    flow_weight_mtx = cosine_similarity(np.array(node_weight))
    np.save('../data.')
    # covert to tensor
    flow_weight_mtx = torch.tensor(flow_weight_mtx, dtype=torch.float)
    return flow_weight_mtx



def flow_similarity():
    flow_mtx = np.load('../data/flow_mtx.npy')
    flow_count_mtx = np.load('../data/flow_count_mtx.npy')

    flow_similarity_matrix = cosine_similarity(flow_mtx)  # OD相似性矩阵
    flow_count_similarity_matrix = cosine_similarity(flow_count_mtx)  # OD count相似性矩阵

    np.save('../data/flow_similarity_mtx.npy', flow_similarity_matrix)
    np.save('../data/flow_count_similarity_mtx.npy', flow_count_similarity_matrix)




if __name__ == '__main__':
    adj_mtx()
    # flow_similarity()








