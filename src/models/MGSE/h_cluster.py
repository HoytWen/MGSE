import numpy as np
import torch as th
import faiss
import torch.nn.functional as F

def run_hkmeans_faiss(features, n_protos, cf, gpus='0'):

    results = {'g2cluster': [], 'centroids': [], 'density': [], 'cluster2cluster': [], 'logits': []}
    tau = cf.student_tau
    for seed, num_proto in enumerate(n_protos):
        # print(seed)
        d = features.shape[1]
        k = int(num_proto)
        clus = faiss.Clustering(d, k)
        clus.verbose = False
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed

        if cf.dataset == 'zinc_standard_agent':
            if seed == 0:
                clus.min_points_per_centroid = 10000
                clus.max_points_per_centroid = 500000
            else:
                clus.min_points_per_centroid = 1
                clus.max_points_per_centroid = 5
        else:
            clus.min_points_per_centroid = 2
            clus.max_points_per_centroid = 1000

        device = th.device("cuda:0" if gpus != '-1' and th.cuda.is_available() else "cpu")
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = int(gpus)
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        if seed == 0:
            clus.train(features, index)
            D, I = index.search(features, 1)
        else:
            clus.train(results['centroids'][seed - 1].cpu().numpy(), index)
            D, I = index.search(results['centroids'][seed - 1].cpu().numpy(), 1)

        g2cluster = [int(n[0]) for n in I]
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(g2cluster):
            Dcluster[i].append(D[im][0])

        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        if seed > 0:
            g2cluster = np.array(g2cluster)  # enable batch indexing
            results['cluster2cluster'].append(th.LongTensor(g2cluster).to(device))
            g2cluster = g2cluster[results['g2cluster'][seed - 1].cpu().numpy()]
            g2cluster = list(g2cluster)

        if len(set(g2cluster)) == 1:
            print("Warning! All samples are assigned to one cluster")

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
        density = tau * density / density.mean()

        # convert to cuda Tensors for broadcast
        centroids = th.Tensor(centroids).to(device)
        centroids = th.nn.functional.normalize(centroids, p=2, dim=1)
        if seed > 0:
            proto_logits = th.mm(results['centroids'][-1], centroids.t())
            results['logits'].append(proto_logits.to(device))

        density = th.Tensor(density).to(device)
        g2cluster = th.LongTensor(g2cluster).to(device)
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['g2cluster'].append(g2cluster)

    return results


def run_hkmeans(model, loader, n_protos, cf):

    results = {'centroids': [], 'cluster2cluster': [], 'centroid_state': []}

    for i, num_proto in enumerate(n_protos):
        if cf.JK == 'concat':
            results['centroids'].append(th.rand(num_proto, (len(n_protos) + 1) * cf.emb_dim).to(cf.compute_dev))
        else:
            results['centroids'].append(th.rand(num_proto, cf.emb_dim).to(cf.compute_dev))

        results['centroid_state'].append(th.zeros(num_proto).to(cf.compute_dev))
        if i < (len(n_protos) - 1):
            results['cluster2cluster'].append(th.zeros(num_proto, dtype=th.long).to(cf.compute_dev))

    model.eval()
    for _ in range(5):
        for step, data in enumerate(loader):
            data = data.to(cf.compute_dev)
            x = model.encoder(data.x, data.edge_index, data.edge_attr)
            g = model.pretrain_pool(x, data.batch)
            g = model.proj(g)

            g_norm = F.normalize(g)
            p_norm = F.normalize(results['centroids'][0])

            sim = th.mm(g_norm, p_norm.t())
            mask = (sim == sim.max(1)[0].unsqueeze(-1)).float()
            cnt = mask.sum(0)
            results['centroid_state'][0].data = results['centroid_state'][0].data + cnt.data

    idx = th.nonzero((results['centroid_state'][0] >= 2).float()).squeeze(-1)
    proto_selected = th.index_select(results['centroids'][0], 0, idx)
    proto_selected.requires_grad = True

    return results