import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
from evaluation import Evaluation
from evaluation_quad import Evaluation_quad
from tqdm import tqdm
from scipy.sparse import coo_matrix
#from deepwalk import Node2Vec
#from torch_geometric.utils import from_scipy_sparse_matrix
#import networkx as nx
import matplotlib.pyplot as plt

# parse settings
setting = Setting()
setting.parse()
dir_name = os.path.dirname(setting.log_file)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
setting.log_file = setting.log_file + '_' + timestring
log = open(setting.log_file, 'w')

# print(setting)

# log_string(log, 'log_file: ' + setting.log_file)
# log_string(log, 'user_file: ' + setting.trans_user_file)
# log_string(log, 'loc_temporal_file: ' + setting.trans_loc_file)
# log_string(log, 'loc_spatial_file: ' + setting.trans_loc_spatial_file)
# log_string(log, 'interact_file: ' + setting.trans_interact_file)

# log_string(log, str(setting.lambda_user))
# log_string(log, str(setting.lambda_loc))

# log_string(log, 'W in AXW: ' + str(setting.use_weight))
# log_string(log, 'GCN in user: ' + str(setting.use_graph_user))
# log_string(log, 'spatial graph: ' + str(setting.use_spatial_graph))

message = ''.join([f'{k}: {v}\n' for k, v in vars(setting).items()])
log_string(log, message)

# load dataset
poi_loader = PoiDataloader(
    setting, setting.max_users, setting.min_checkins)  # 0， 5*20+1
poi_loader.read(setting.dataset_file)
poi_loader.create_csr_matrix()
# print('Active POI number: ', poi_loader.locations())  # 18737 106994
# print('Active User number: ', poi_loader.user_count())  # 32510 7768
# print('Total Checkins number: ', poi_loader.checkins_count())  # 1278274

log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))
log_string(log, 'Total Quadkey number:{}'.format(poi_loader.quadkey_count()))
log_string(log, 'Total Time number:{}'.format(poi_loader.time_count()))

# create flashback trainer
with open(setting.trans_loc_file, 'rb') as f:  # transition POI graph
    transition_graph = pickle.load(f)  # 在cpu上
# transition_graph = top_transition_graph(transition_graph)
transition_graph = coo_matrix(transition_graph)

if setting.use_spatial_graph:
    with open(setting.trans_loc_spatial_file, 'rb') as f:  # spatial POI graph
        spatial_graph = pickle.load(f)  # 在cpu上
    # spatial_graph = top_transition_graph(spatial_graph)
    spatial_graph = coo_matrix(spatial_graph)
else:
    spatial_graph = None

if setting.use_graph_user:
    with open(setting.trans_user_file, 'rb') as f:
        friend_graph = pickle.load(f)  # 在cpu上
    # friend_graph = top_transition_graph(friend_graph)
    friend_graph = coo_matrix(friend_graph)
else:
    friend_graph = None

with open(setting.trans_interact_file, 'rb') as f:  # User-POI interaction graph
    interact_graph = pickle.load(f)  # 在cpu上
interact_graph = csr_matrix(interact_graph)

log_string(log, 'Successfully load graph')

dataset = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TRAIN)  # 20, 200 or 1024, 0
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(
), 'batch size must be lower than the amount of available users'


trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s, setting.lambda_loc, setting.lambda_user,
                           setting.use_weight, transition_graph, spatial_graph, friend_graph, setting.use_graph_user,
                           setting.use_spatial_graph, interact_graph)  # 0.01, 100 or 1000
h0_strategy = create_h0_strategy(
    setting.hidden_dim, setting.is_lstm)  # 10 True or False
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), poi_loader.quadkey_count(), poi_loader.quadkey_location(), poi_loader.time_count(), setting.hidden_dim, setting.rnn_factory,
                setting.device, setting)
evaluation_test = Evaluation(dataset_test, dataloader_test,
                             poi_loader.user_count(), h0_strategy, trainer, setting, log)
evaluation_test_quad = Evaluation_quad(dataset_test, dataloader_test,
                             poi_loader.user_count(), h0_strategy, trainer, setting, log)
print('{} {}'.format(trainer, setting.rnn_factory))

#  training loop
optimizer = torch.optim.Adam(trainer.parameters(
), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20, 40, 60, 80], gamma=0.2)

param_count = trainer.count_parameters()
log_string(log, f'In total: {param_count} trainable parameters')


evaluation_test.evaluate()
#evaluation_test_quad.evaluate()

bar = tqdm(total=setting.epochs)
bar.set_description('Training')

for e in range(setting.epochs):  # 100
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    dataset.shuffle_users()  # shuffle users before each epoch!

    losses = []
    epoch_start = time.time()
    for i, (x, t, w, d, t_slot, hour, s, q, y, y_t, y_w, y_d, y_t_slot, y_hour, y_s, y_q, reset_h, active_users) in enumerate(dataloader):
        # reset hidden states for newly added users
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])

        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        w = w.squeeze().to(setting.device)
        d = d.squeeze().to(setting.device)
        t_slot = t_slot.squeeze().to(setting.device)
        hour = hour.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)
        q = q.squeeze().to(setting.device)

        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_w = y_w.squeeze().to(setting.device)
        y_d = y_d.squeeze().to(setting.device)
        y_t_slot = y_t_slot.squeeze().to(setting.device)
        y_hour = y_hour.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)
        y_q = y_q.squeeze().to(setting.device)#[..., -1]
        active_users = active_users.to(setting.device)
        # print(q)
        optimizer.zero_grad()
        loss = trainer.loss(x, q, t, w, d, t_slot, hour, s, y, y_t, y_w, y_d,
                            y_t_slot, y_hour, y_s, y_q, h, active_users, e)

        loss2=trainer.model.Loss_l2()*1.5e-6
        lossx=loss+loss2
        lossx.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(trainer.parameters(), 5)   #5
        losses.append(lossx.item())
        optimizer.step()

    # schedule learning rate:
    scheduler.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One training need {:.2f}s'.format(
        epoch_end - epoch_start))
    # statistics:
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        log_string(log, f'Epoch: {e + 1}/{setting.epochs}')
        log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')

    if (e + 1) % setting.validate_epoch == 0:
    # if (e+1)%5==0 and (e+1)>=35 and (e+1)!=30:
        log_string(log, f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
        evl_start = time.time()
        evaluation_test.evaluate()
        #evaluation_test_quad.evaluate()
        evl_end = time.time()
        log_string(log, 'One evaluate need {:.2f}s'.format(
            evl_end - evl_start))

bar.close()

# print("load parameters")
# trainer.save_parameters()
# print("load parameters successful")