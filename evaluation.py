import torch
from utils import get_rank, get_metrics
from tqdm import tqdm

def evaluate(my_model, target, init_emb_ent, init_emb_rel, relation_triplets, qual=False):
    with torch.no_grad():
        my_model.eval()
        msg = torch.tensor(target.msg_triplets).cuda()
        sup = torch.tensor(target.sup_triplets).cuda()

        emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)

        head_ranks = []
        tail_ranks = []
        ranks = []
        for triplet in tqdm(sup):
            triplet = triplet.unsqueeze(dim = 0)
            
            if not qual:
                head_corrupt = triplet.repeat(target.num_ent, 1)
                head_corrupt[:,0] = torch.arange(end = target.num_ent)
                
                head_scores = my_model.score(emb_ent, emb_rel, head_corrupt)
                head_filters = target.filter_dict[('_', int(triplet[0,1].item()), int(triplet[0,2].item()))]
                head_rank = get_rank(triplet, head_scores, head_filters, target = 0)
                
                ranks.append(head_rank)
                head_ranks.append(head_rank)

            tail_corrupt = triplet.repeat(target.num_ent, 1)
            tail_corrupt[:,2] = torch.arange(end = target.num_ent)

            tail_scores = my_model.score(emb_ent, emb_rel, tail_corrupt)
            tail_filters = target.filter_dict[(int(triplet[0,0].item()), int(triplet[0,1].item()), '_')]
            tail_rank = get_rank(triplet, tail_scores, tail_filters, target = 2)

            ranks.append(tail_rank)
            tail_ranks.append(tail_rank)

        return ranks
