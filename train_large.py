import csv
import random
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_sparse import SparseTensor

from src.batch_sampler import NeighborSampler

from src.utils import setup_seed, get_mask, clustering
from src.sim_model import Model, Encoder
from src.clustering_module import DEC_Clustering
from src.clustering_metrics import clustering_metrics
import src.plot_clusters as plot
import torch.serialization
from torch_geometric.data.data import Data, DataEdgeAttr


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--max_duration', type=int,
                    default=60, help='max duration time')
parser.add_argument('--kmeans_device', type=str,
                    default='cpu', help='kmeans device, cuda or cpu')
parser.add_argument('--kmeans_batch', type=int, default=-1,
                    help='batch size of kmeans on GPU, -1 means full batch')
parser.add_argument('--batchsize', type=int, default=2048, help='')

# dataset para
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')

# model para
parser.add_argument('--hidden_channels', type=str, default='512,256')
parser.add_argument('--size', type=str, default='10,10', help='')
parser.add_argument('--projection', type=str, default='')
parser.add_argument('--tau', type=float, default=0.5, help='temperature')
parser.add_argument('--ns', type=float, default=0.5)

# sample para
parser.add_argument('--wt', type=int, default=20)
parser.add_argument('--wl', type=int, default=4)
parser.add_argument('--n', type=int, default=2048)

# learning para
parser.add_argument('--dropout', type=float, default=0, help='')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--epochs_sim', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.5, help='')
parser.add_argument('--epochs_cluster', type=int, default=400)

args = parser.parse_args()


def train():
    ts = time.time()
    randint = random.randint(1, 1000000)
    setup_seed(randint)
    if args.verbose:
        print('random seed : ', randint, '\n', args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
      path = './data/OGB/'
      from ogb.nodeproppred import PygNodePropPredDataset
      from  torch import serialization
      
      # Add the safe globals before loading
      from torch_geometric.data.data import DataEdgeAttr
      torch.serialization.add_safe_globals([DataEdgeAttr])
      
      dataset = PygNodePropPredDataset(root=path, name=args.dataset)
      x, edge_index, y = dataset[0].x, dataset[0].edge_index, dataset[0].y
      y = y[:, 0]

    elif args.dataset == 'Reddit':
        path = './data/Reddit/'
        dataset = Reddit(root=path)
        x, edge_index, y = dataset[0].x, dataset[0].edge_index, dataset[0].y
    else:
        raise RuntimeError(f"Unknown dataset {args.dataset}")

    edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])

    N, E, num_features = x.shape[0], edge_index.shape[-1], x.shape[-1]
    print(f"Loading {args.dataset} is over, num_nodes: {N: d}, num_edges: {E: d}, "
          f"num_feats: {num_features: d}, time costs: {time.time()-ts: .2f}")

    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1], sparse_sizes=(N, N))
    adj.fill_value_(1.)

    hidden = list(map(int, args.hidden_channels.split(',')))
    if args.projection == '':
        projection = None
    else:
        projection = list(map(int, args.projection.split(',')))
    size = list(map(int, args.size.split(',')))
    # print(len(hidden), len(size))
    assert len(hidden) == len(size)

    train_loader = NeighborSampler(edge_index, adj,
                                   is_train=True,
                                   node_idx=None,
                                   wt=args.wt,
                                   wl=args.wl,
                                   sizes=size,
                                   batch_size=args.batchsize,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=6)

    test_loader = NeighborSampler(edge_index, adj,
                                  is_train=False,
                                  node_idx=None,
                                  sizes=size,
                                  batch_size=10000,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=6)

    encoder = Encoder(num_features, hidden_channels=hidden,
                      dropout=args.dropout, ns=args.ns).to(device)
    model = Model(
        encoder, in_channels=hidden[-1], project_hidden=projection, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    dataset2n_clusters = {'ogbn-arxiv': 40, 'Reddit': 41,
                          'ogbn-products': 47, 'ogbn-papers100M': 172}
    n_clusters = dataset2n_clusters[args.dataset]

    x = x.to(device)
    print(f"Start training")

    ts_train = time.time()
    stop_pos = False
    for epoch in range(1, args.epochs_sim):
        model.train()
        total_loss = total_examples = 0

        for (batch_size, n_id, adjs), adj_batch, batch in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            if len(hidden) == 1:
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]

            adj_ = get_mask(adj_batch)
            optimizer.zero_grad()
            out = model(x[n_id].to(device), adjs=adjs)
            out = F.normalize(out, p=2, dim=1)
            loss = model.loss(out, adj_)

            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            total_examples += batch_size

            if args.verbose:
                print(f'(T) | Epoch {epoch:02d}, loss: {loss:.4f}, '
                      f'train_time_cost: {time.time() - ts_train:.2f}, examples: {batch_size:d}')

            train_time_cost = time.time() - ts_train
            if train_time_cost // 60 >= args.max_duration:
                print(
                    "*********************** Maximum training time is exceeded ***********************")
                stop_pos = True
                break
        if stop_pos:
            break

    print(f'Finish training, training time cost: {time.time() - ts_train:.2f}')

    with torch.no_grad():
        model.eval()
        z = []
        for count, ((batch_size, n_id, adjs), _, batch) in enumerate(tqdm(test_loader)):
            if len(hidden) == 1:
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]
            out = model(x[n_id].to(device), adjs=adjs)
            z.append(out.detach().cpu().float())
        z = torch.cat(z, dim=0)
        z = F.normalize(z, p=2, dim=1)

    dec = DEC_Clustering(input_dim=out.shape[1], n_clusters=n_clusters, alpha=args.alpha).to(device)

    if out.device != device:
        out = out.to(device)

    dec.initialize_clusters(out)
    
    # 4. Setup optimizer
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(1, args.epochs_cluster):
        assignments, pooled, kl_loss, recon_loss, total_loss, final_emb = dec(out)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss={total_loss.item():.4f} "
                f"(KL={kl_loss.item():.4f} Recon={recon_loss.item():.4f})")

    # In the evaluation section of train():
    with torch.amp.autocast(device_type='cuda'):  # Updated autocast
        final_assignments, _, _, _, _, final_embd = dec(out)
        cluster_ids = final_assignments.argmax(dim=1)

    print(final_assignments)
    print(cluster_ids)
    plot.plot(final_embd, y, "after similarity", args.dataset)

    # Convert to numpy properly
    metrics_eval = clustering_metrics(y.cpu().numpy(), cluster_ids.cpu().numpy())  # Added .cpu()

    acc, nmi, ari, fms, f1_macro, f1_micro = metrics_eval.evaluationClusterModelFromLabel(tqdm=None)
    print("clusters: ", len(np.unique(y.cpu().numpy())), len(np.unique(cluster_ids.cpu().numpy())))
  
    print(
        f'train over | ACC={acc:.4f}, NMI={nmi:.4f},  ARI={ari:.4f}, f1_macro={f1_macro:.4f}, f1_micro={f1_micro:.4f}')

    return acc, nmi, ari, f1_macro, f1_micro

def run(runs=1, result=None):
    if result:
        with open(result, 'w', encoding='utf-8-sig', newline='') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(
                ['runs', 'acc', 'nmi', 'ari', 'f1_macro', 'f1_micro'])

    ACC, NMI, ARI, F1_MA, F1_MI = [], [], [], [], []
    for i in range(runs):
        print(f'\n----------------------runs {i+1: d} start')
        acc, nmi, adjscore, f1_macro, f1_micro = train()
        print(f'\n----------------------runs {i + 1: d} over')
        if result:
            with open(result, 'a', encoding='utf-8-sig', newline='') as f_w:
                writer = csv.writer(f_w)
                writer.writerow([i+1, acc, nmi, adjscore, f1_macro, f1_micro])

        ACC.append(acc)
        NMI.append(nmi)
        ARI.append(adjscore)
        F1_MA.append(f1_macro)
        F1_MI.append(f1_micro)

    ACC = np.array(ACC)
    NMI = np.array(NMI)
    ARI = np.array(ARI)
    F1_MA = np.array(F1_MA)
    F1_MI = np.array(F1_MI)
    if result:
        with open(result, 'a', encoding='utf-8-sig', newline='') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(['mean', ACC.mean(), NMI.mean(),
                            ARI.mean(), F1_MA.mean(), F1_MI.mean()])
            writer.writerow(['std', ACC.std(), NMI.std(),
                            ARI.std(), F1_MA.std(), F1_MI.std()])


if __name__ == '__main__':
    result = None
    run(args.runs, result)
