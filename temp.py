import torch
from tqdm import tqdm
from model import InGram
from utils import initialize, generate_neg
from dataset import TrainData, TestNewData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print("device: ", torch.cuda.get_device_name(0))

train = TrainData("./data_dummy/dummy", qual=False)
msg, sup = train.split_transductive(0.75)

print("Get training data")
print(msg)
print("\n")
print(sup)
print("\n")

valid = TestNewData("./data_dummy/dummy", qual=True, data_type="valid")
msg = torch.tensor(valid.msg_triplets).cuda()
sup = torch.tensor(valid.sup_triplets).cuda()
print("Get validation data")
print(msg)
print("\n")
print(sup)
print("\n")


# print("Run training")
# d_e = 32
# d_r = 32
# hdr_e = 8
# hdr_r = 4
# B = 10
# num_neg = 10
# lr = 5e-4

# my_model = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r, \
# 				num_bin = B, num_layer_ent = 2, num_layer_rel = 2, \
# 				num_head = 8)
# my_model = my_model.cuda()
# loss_fn = torch.nn.MarginRankingLoss(margin = 2, reduction = 'mean')

# optimizer = torch.optim.Adam(my_model.parameters(), lr = lr)
# epochs = 1000
# pbar = tqdm(range(epochs))
# losses = []

# for epoch in pbar:
# 	optimizer.zero_grad()
# 	msg, sup = train.split_transductive(0.75)

# 	init_emb_ent, init_emb_rel, relation_triplets = initialize(train, msg, d_e, d_r, B)
# 	msg = torch.tensor(msg).cuda()
# 	sup = torch.tensor(sup).cuda()

# 	emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)
# 	pos_scores = my_model.score(emb_ent, emb_rel, sup)
# 	neg_scores = my_model.score(emb_ent, emb_rel, generate_neg(sup, train.num_ent, num_neg = num_neg))

# 	loss = loss_fn(pos_scores.repeat(num_neg), neg_scores, torch.ones_like(neg_scores))
# 	loss.backward()
# 	torch.nn.utils.clip_grad_norm_(my_model.parameters(), 0.1, error_if_nonfinite = False)
# 	optimizer.step()
# 	pbar.set_description(f"loss {loss.item()}")
# 	losses.append(loss.item())

# for loss in losses:
#       print(loss)
