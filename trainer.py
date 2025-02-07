import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from utils import *
from network import Flashback
from scipy.sparse import csr_matrix


class FlashbackTrainer():
    """ Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    """

    def __init__(self, lambda_t, lambda_s, lambda_loc, lambda_user, use_weight, transition_graph, spatial_graph,
                 friend_graph, use_graph_user, use_spatial_graph, interact_graph):
        """ The hyper parameters to control spatial and temporal decay.
        """
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.graph = transition_graph
        self.spatial_graph = spatial_graph
        self.friend_graph = friend_graph
        self.interact_graph = interact_graph

    def __str__(self):
        return 'Use flashback training.'

    def count_parameters(self):
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count += param.numel()
        return param_count
    
    def parameters(self):
        return self.model.parameters()
    
    def save_parameters(self):
        torch.save(self.model.state_dict(), "gowalla.pt")
    
    def load_parameters(self):
        self.model.load_state_dict(torch.load("gowalla.pt", map_location={'cuda:0':'cuda:0'}), strict=False) #, map_location={'cuda:0':'cuda:1'}
    

    def prepare(self, loc_count, user_count, quadkey_count, quadkey_location, time_count, hidden_size, gru_factory, device, setting):
        def f_t(delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay

        # exp decay  2个functions
        def f_s(delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))
        self.loc_count = loc_count + 1 
        self.quadkey_count = quadkey_count
        self.time_count = time_count
        self.quadkey_location = quadkey_location
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cross_entropy_loss1 = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.mseloss = nn.MSELoss()
        self.model = Flashback(self.loc_count, user_count, quadkey_count, quadkey_location, time_count, hidden_size, f_t, f_s, gru_factory, self.lambda_loc,
                               self.lambda_user, self.use_weight, self.graph, self.spatial_graph, self.friend_graph,
                               self.use_graph_user, self.use_spatial_graph, self.interact_graph, setting).to(device)

    def evaluate(self, x, q, t, w, d, t_slot, hour, s, y_t, y_t_slot, y_s, h, active_users):
        """ takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction!
        """

        self.model.eval()
        # (seq_len, user_len, loc_count)
        out, q_out, t_out, h, q_1, q_2, location_topk, filter_result, retain_result, kl_out = self.model(x, q, t, w, d, t_slot, hour, s, y_t,
                            y_t_slot, y_s, h, active_users, 0, state="eval")

        out_t = out.transpose(0, 1)
        return out_t, q_out, h, location_topk.transpose(0, 1), filter_result.transpose(0, 1), retain_result.transpose(0, 1)  # model outputs logits

    def loss(self, x, q, t, w, d, t_slot, hour, s, y, y_t, y_w, y_d, y_t_slot, y_hour, y_s, y_q, h, active_users, epoch):
        """ takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss """

        self.model.train()
        out, q_out, t_out, h, q_1, q_2, location_topk, filter_result, retain_result, kl_out = self.model(x, q, t, w, d, t_slot, hour, s, y_t, y_t_slot, y_s, h,
                            active_users, epoch, state="train")  # out (seq_len, batch_size, loc_count)
        out = out.contiguous().view(-1, self.loc_count)  # (seq_len * batch_size, loc_count)
        y = y.view(-1)  # (seq_len * batch_size)
        # kl_out = kl_out.view(-1, self.loc_count)

        # q_kl_1 = torch.softmax(q_1, dim=-1)
        # q_kl_2 = torch.softmax(q_2, dim=-1)
        # consistency_loss = F.kl_div(q_kl_1.log(), q_kl_2, reduction='batchmean')
        # out_softmax = torch.softmax(out, dim=-1)
        # kl_out = torch.softmax(kl_out, dim=-1)
        # kl_loss = F.kl_div(out_softmax.log(), kl_out, reduction='batchmean')

        quad_loss = []
        quad_loss2 = []
        for index, quadkey_num in enumerate(self.quadkey_count[:-1]):
            q_output = q_out[index].contiguous().reshape(-1, quadkey_num)
            #q_output2 = q_2[index].contiguous().reshape(-1, quadkey_num)
            y_quad = y_q[..., index].view(-1)
            quad_loss.append(self.cross_entropy_loss1(q_output, y_quad))
            #quad_loss2.append(self.cross_entropy_loss1(q_output2, y_quad))
        
        y_times = torch.stack([y_w, y_d, y_t_slot, y_hour], dim=-1)
        time_loss = []
        for index, time_num in enumerate(self.time_count[:-1]):
            t_output = t_out[index].contiguous().reshape(-1, time_num)
            y_time = y_times[..., index].view(-1).long()
            time_loss.append(self.cross_entropy_loss1(t_output, y_time))

        
        l = self.nll(out, y) + 0.5 * sum(quad_loss) + 0.5 * sum(time_loss) #+ 0.5 * sum(quad_loss2) # + consistency_loss + kl_loss #self.mseloss(q_out, y_q.transpose(0,1).float()) #self.cross_entropy_loss(q_out, y_q)  #+ self.mseloss(q_out, y_q.transpose(0,1).float())
        return l
