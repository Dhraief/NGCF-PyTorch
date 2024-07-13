from torch_geometric.data import Data
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import random



class SenderToReceiverData(Data):
    """
    A data object representing a sender-to-receiver bipartite graph.
    node_idx: node indices (senders + receivers)
    senders: node indices of senders
    receivers: node indices of receivers
    edge_index: (relabeled) edge indices (senders -> receivers)
    num_nodes: number of nodes (senders + receivers)
    y: label
    """

    @staticmethod
    def from_data(senders: torch.Tensor, receivers: torch.Tensor, y: torch.Tensor):
        num_senders = senders.size(0)
        num_receivers = receivers.size(0)
        edge_index = torch.stack(
            torch.meshgrid(
                torch.arange(num_senders),
                torch.arange(num_senders, num_senders + num_receivers),
                indexing="ij",
            )
        ).reshape(2, -1)
        return SenderToReceiverData(
            node_idx=torch.cat([senders, receivers]),
            senders=senders,
            receivers=receivers,
            edge_index=edge_index,
            num_nodes=num_senders + num_receivers,
            y=y,
        )
    

class EllipticRecommendationEvalDataset(Dataset):
    def __init__(self, data_generator, args):
        self.num_illicits = args.a
        self.num_licits = args.b
        self.num_samples = args.num_samples
        self.data_generator = data_generator
        self.args = args
        self._generate_data()
        self.device = args.device

    def _generate_data(self):
        self.data_list = []
        for _ in tqdm(range(self.num_samples), desc="Generating data"):
            self.data_list.append(self._generate_sample())
        
    def _generate_sample(self):
        
        chosen_illicit = random.sample(self.data_generator.data_illicit, self.num_illicits)
        chosen_licit = random.sample(self.data_generator.data_licit, self.num_licits)


        illicit_senders, illicit_receivers = [],[]
        for s,r in chosen_illicit:
            illicit_senders.append(s)
            illicit_receivers.append(r)
        receivers, senders = illicit_receivers, illicit_senders
        illicit_receivers, illicit_senders = torch.tensor(illicit_receivers), torch.tensor(illicit_senders)
        for r,s in chosen_licit:
            # receivers.append(r)
            senders.append(s)
        illicit_edge_index = torch.stack(
            [illicit_senders, illicit_receivers], dim=0
        ).contiguous()
        senders = torch.tensor(list(senders))
        receivers = torch.tensor(list(receivers))
        return senders, receivers, illicit_edge_index
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_list[idx]


        
        