from utility.evaluator import EdgeRecommendationEvaluator
from torch_geometric.data import Batch
import torch
from utility.eval_utils.data import *
from  torch.utils.data import DataLoader 
from einops import einsum


def eval_elliptic(data_generator, model, args):
    eval_dataset = EllipticRecommendationEvalDataset(data_generator, args) # construct_eval_dataset [(a illicit, b licits)* n_samples], senders, receivers, illicit_edge_indices = batch
    for s,r,edge_index in eval_dataset.data_list:
        #assert r.shape[0] == eval_dataset.num_illicits, f"Expected {eval_dataset.num_illicits} illicit senders, got {s.shape[0]}"
        assert s.shape[0] == eval_dataset.num_licits+eval_dataset.num_illicits, f"Expected {eval_dataset.num_licits} licit receivers, got {r.shape[0]}"
        assert edge_index.shape == (2, eval_dataset.num_illicits), f"Expected edge_index shape {(2, eval_dataset.num_illicits)}, got {edge_index.shape}"
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False) # DataLoader
    evaluator = EdgeRecommendationEvaluator()


    top_k_edges = []
    hr_list, ndcg_list = [], []
    for batch in eval_loader:
        senders, receivers, illicit_edge_indices = batch
        #print("Shapes", len(senders), len(receivers), illicit_edge_indices.shape)
        batch_size = len(senders)
        new_batch = Batch.from_data_list(
            [
                SenderToReceiverData.from_data(s, r, torch.tensor([1]))
                for s, r in zip(senders, receivers)
            ],
            follow_batch=["senders", "receivers"],
        ).to(model.device)

        senders, receivers, senders_batch, receivers_batch = (
            new_batch.senders,
            new_batch.receivers,
            new_batch.senders_batch,
            new_batch.receivers_batch,
        )
        receiver_features, sender_features, _ = model(list(receivers),list(senders),[])
        top_k_edges = []
        debug = False
        for i in range(batch_size):
            curr_top_k_edges = []
            assert senders.device == senders_batch.device
            assert len(senders) == len(senders_batch)
                        # Debugging statements
            if debug:
                print(f"Processing batch {i}")
                print(f"senders: {senders}")
                print(f"senders_batch: {senders_batch}")


            curr_senders = senders[senders_batch == i]
            if debug:
                print(f"receivers: {receivers}")
                print(f"receivers_batch: {receivers_batch}")
                
            curr_receivers = receivers[receivers_batch == i]
            curr_sender_features = sender_features[senders_batch == i]
            curr_receiver_features = receiver_features[receivers_batch == i]
            # compute dot product for all pairs of senders and receivers
            scores = einsum(
                curr_sender_features, curr_receiver_features, "i d, j d -> i j"
            )
            # get top k edges
            top_k_indices = torch.topk(
                scores.flatten(), min(args.k, scores.size(0) * scores.size(1))
            ).indices
            top_k_senders = curr_senders[top_k_indices // scores.size(1)]
            top_k_receivers = curr_receivers[top_k_indices % scores.size(1)]
            for s, r in zip(top_k_senders, top_k_receivers):
                curr_top_k_edges.append((s.item(), r.item()))
            top_k_edges.append(curr_top_k_edges)
        hit_ratio, ndcg = evaluator(top_k_edges, illicit_edge_indices)
        hr_list.append(hit_ratio)
        ndcg_list.append(ndcg)

    #print("Hit ratio", hr_list)
    #print("NDCG", ndcg_list)
    hit_ratio = sum(hr_list) / len(hr_list)
    ndcg = sum(ndcg_list) / len(ndcg_list)
    return hit_ratio, ndcg

