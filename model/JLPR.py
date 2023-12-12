import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric.utils as utils
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, dense_diff_pool, dense_mincut_pool
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# torch.manual_seed(15)

def compute_metrics(y_pred, y_test):
    y_pred[y_pred<0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    epsilon = np.finfo(np.float64).eps
    mape = np.mean(np.abs(y_pred - y_test) / np.maximum(np.abs(y_test), epsilon)) * 100
    return mae, np.sqrt(mse), r2, mape

def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return y_pred

def kf_predict(X, Y):
    kf = KFold(n_splits=len(X))
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, y_train, X_test, 1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds), np.concatenate(y_truths)


def predict_regression(embs, labels, display=False):
    y_pred, y_test = kf_predict(embs, labels)
    mae, rmse, r2, mape = compute_metrics(y_pred, y_test)
    if display:
        print("MAE:{}, RMSE:{}, r2:{}, mape:{}".format(mae, rmse, r2, mape))
    return mae, rmse, r2, mape, y_pred, y_test

def do_task(crime_label, embs, epoch, display=True):
    crime_mae, crime_rmse, crime_r2, mape, y_pred, y_test = predict_regression(embs, crime_label, display=display)
    # save predicted values
    with open('./output_net4/8. newtest_crime_price/pred_truth_e{}.txt'.format(epoch), 'w') as f:
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(y_pred[i], y_test[i]))
        f.close()

    return crime_mae, crime_rmse, crime_r2, mape


class AssignmentMatrix(nn.Module):  # FC
    def __init__(self, input, ncluster):
        super().__init__()
        self.assign_mat1 = nn.Linear(input, 2 * input)
        self.assign_mat2 = nn.Linear(2 * input, ncluster)

    def forward(self, x):
        x = F.relu(self.assign_mat1(x))
        s = self.assign_mat2(x)
        s = F.softmax(s, dim=0)
        return s

class Net(nn.Module):
    def __init__(self, in_dim, hidd_dim, n_cluster, out_dim):
        super().__init__()
        self.gnn1 = GCNConv(in_dim, 256)
        self.gnn2 = GCNConv(256, hidd_dim)

        self.assign_mat = AssignmentMatrix(hidd_dim, n_cluster)

        self.gnn3 = GCNConv(hidd_dim, 2 * hidd_dim)
        self.gnn4 = GCNConv(2 * hidd_dim, out_dim)

        self.pred1 = nn.Linear(out_dim, 2 * out_dim)
        self.pred2 = nn.Linear(2 * out_dim, 2 * out_dim)
        self.pred3 = nn.Linear(2 * out_dim, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.gnn1(x, edge_index, edge_weight))
        x = self.gnn2(x, edge_index, edge_weight)

        assign_mtx = self.assign_mat(x)

        map_matrix = (assign_mtx == torch.max(assign_mtx, dim=-1, keepdim=True)[0]).float()

        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)

        region, region_adj, loss1, loss2 = just_balance_pool(x, adj, map_matrix)
        # region, region_adj, loss1, loss2 = mincut_pool(x, adj, map_matrix)

        region, region_adj = region.squeeze(0), region_adj.squeeze(0)

        region_adj = region_adj.to_sparse_coo()
        region_adj_edge_index, region_adj_edge_weight = region_adj.indices(), region_adj.values()

        x_region = F.relu(self.gnn3(region, region_adj_edge_index, region_adj_edge_weight))
        x_region = self.gnn4(x_region, region_adj_edge_index, region_adj_edge_weight)

        pred = F.relu(self.pred1(x_region))
        pred = F.relu(self.pred2(pred))
        pred = self.pred3(pred)

        return map_matrix, loss1, loss2, x_region, pred

def mincut_pool(x, adj, s, mask=None):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    # d = torch.einsum('ijk->ij', out_adj)
    # d = torch.sqrt(d)[:, None] + EPS
    # out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss


def just_balance_pool(x, adj, s, mask=None, normalize=True):
    EPS = 1e-15
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    # s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Loss
    ss = torch.matmul(s.transpose(1, 2), s)
    ss_sqrt = torch.sqrt(ss + EPS)
    loss = torch.mean(-_rank3_trace(ss_sqrt))
    if normalize:
        loss = loss / torch.sqrt(torch.tensor(num_nodes * k))

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    # d = torch.einsum('ijk->ij', out_adj)
    # d = torch.sqrt(d)[:, None] + EPS
    # out_adj = (out_adj / d) / d.transpose(1, 2)

    ent_loss = (-s * torch.log(s + 1e-6)).sum(dim=-1).mean()

    return out, out_adj, loss, ent_loss

def _rank3_trace(x):
    return torch.einsum('ijj->i', x)

def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out

def write_emb_assign(epoch, ncluster, emb, assign):
    ffolder = './output_net4/m_{}/'.format(ncluster)
    if not os.path.exists(ffolder):
        os.makedirs(ffolder)

    emb_path = ffolder + 'region_emb_e' + str(epoch)
    assign_path = ffolder + 'assign_' + str(epoch)

    np.save(emb_path, emb)
    np.save(assign_path, assign)


def write_metrics(epoch, ncluster, mae, rmse, r2, mape, s_num):
    fname = './output_net4/metrics_m_{}.txt'.format(ncluster)
    f = open(fname, 'a+')
    content = str(epoch) + ',' + str(mae) + ',' + str(rmse) + ',' + str(r2) + ',' + str(mape) + ',' + str(s_num) + '\n'
    f.write(content)
    f.close()




adj = np.load('../data/adj_mtx1.npy')
flow = np.load('../data/flow_count_mtx.npy')
flow[flow < 50] = 0.0
f_norm = (flow - flow.min()) / (flow.max() - flow.min())

poi = np.load('../data/poi_emb1_mtx.npy')
img = np.load('../data/img_emb_mtx.npy')
crime = np.load('../data/crime.npy')
price = np.load('../data/train_test_average_house_price.npy')


target = np.concatenate((crime, price), axis=1)

feat = np.concatenate((poi, img), axis=1)  #(801, 160)
fear_all = np.concatenate((feat, crime), axis=1)

alpha = 0.75

X = torch.tensor(feat, dtype=torch.float)
Y = torch.tensor(target, dtype=torch.float)
A = torch.tensor(adj, dtype=torch.float)
f_norm = torch.tensor(f_norm, dtype=torch.float)

A_new = alpha * A + (1-alpha) * f_norm
A = utils.dense_to_sparse(A_new)

edge_index = A[0]
edge_weight = A[1]
data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=Y)


n_cluster = 100

model = Net(data.num_features, 128, n_cluster, 96)
data = data.to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epoch')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test mae')

losses_train = []
test_mae = []

best_loss = float('inf')
counter = 0

dict_target = {'crime': 0, 'price': 2}

# target_y = dict_target['crime']
target_y = dict_target['price']

for epoch in range(500):
    model.train()
    optimizer.zero_grad()

    s, loss1, loss2, region_emb, pred = model(data)

    s_num = len(torch.unique(s.max(dim=-1, keepdim=True)[1]))
    print('epoch:{}; partition numbers: {}'.format(epoch+1, s_num))

    ytruth = data.y[:, target_y].view(-1) @ s

    ypred = pred.view(-1)

    loss_pred = criterion(ypred, ytruth)

    loss = loss_pred + loss1

    print('loss:{}'.format(loss))

    losses_train.append(loss.item())

    loss.backward()

    optimizer.step()

    # crime
    # region_emb = region_emb.detach().cpu().numpy()
    # ytest = data.y[:, target_y + 1].view(-1) @ s
    # ytest = ytest.cpu()

    # price
    s_count = torch.sum(s, dim=0)
    region_emb = region_emb.detach().cpu().numpy()
    ytest = data.y[:, target_y +1].view(-1) @ s
    ytest = ytest / s_count
    ytest = torch.nan_to_num(ytest)
    ytest = ytest.cpu()

    y_zero_index = [i for i in range(len(ytest)) if ytest[i] == 0.]
    x = np.delete(region_emb, y_zero_index, axis=0)
    y = np.delete(ytest, y_zero_index)

    mae, rmse, r2, mape = do_task(y, x, epoch+1)
    test_mae.append(mae)

    ax1.plot(np.arange(epoch + 1), losses_train, 'r', lw=1)
    ax2.plot(np.arange(epoch + 1), test_mae, 'g', lw=1)
    plt.pause(0.1)

    if mae < best_loss:
        best_loss = mae
        counter = 0

        print(mae, rmse, r2)
        write_emb_assign(epoch + 1, n_cluster, region_emb, s.detach().cpu().numpy())
        write_metrics(epoch + 1, n_cluster, mae, rmse, r2, mape, s_num)
        print()

    else:
        counter += 1
        if counter >= 50:
            print('test loss has not improved for 20 epochs. Training stops.')
            break







