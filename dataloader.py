import os.path
import sys

from datetime import datetime
from dataset import PoiDataset, Usage
from quadkey_encoder import latlng2quadkey
from torchtext.data import Field
from collections import defaultdict
from nltk import ngrams
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import torch
import math

class PoiDataloader():
    """ Creates datasets from our prepared Gowalla/Foursquare data files.
    The file consist of one check-in per line in the following format (tab separated):

    <user-id> <timestamp> <latitude> <longitude> <location-id>

    Check-ins for the same user have to be on continuous lines.
    Ids for users and locations are recreated and continuous from 0.
    """

    def __init__(self, setting, max_users=0, min_checkins=0):
        """ max_users limits the amount of users to load.
        min_checkins discards users with less than this amount of checkins.
        """

        self.max_users = max_users  # 0
        self.min_checkins = min_checkins  # 101

        self.user2id = {}
        self.poi2id = {}
        self.poi2gps = {}  # 自己加的
    

        self.users = []
        self.times = []  # 二重列表,每个元素是active user所对应POIs的访问timestamp,按时间顺序排序
        self.time_slots = []
        self.time_hours = []
        self.time_days = []
        self.time_weeks = []
        self.coords = []  # 二重列表,每个元素是active user所对应POIs的GPS坐标,按时间顺序排序
        self.locs = []  # 二重列表,每个元素是active user所对应POIs,按时间顺序排序
        self.quadkeys = []
        self.all_quadkeys = []
        self.map_level = setting.quadkey_list
        self.map_len = len(setting.quadkey_list)
        self.n_quadkey = [0] * self.map_len
        self.quadkeys_num = []
        self.quadkey_level =[[] for _ in range(len(self.map_level))]
        self.quadkey_dict = {}

        self.quadkey2idx = [{} for _ in range(len(self.map_level))]
        self.idx2quadkey = [{} for _ in range(len(self.map_level))]
        self.quadidx2loc = [{} for _ in range(len(self.map_level))]   #defaultdict(set)
        self.quadidx2level = [{} for _ in range(len(self.map_level)-1)]

    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
        return PoiDataset(self.users.copy(),
                          self.times.copy(),
                          self.time_weeks.copy(),
                          self.time_days.copy(),
                          self.time_slots.copy(),
                          self.time_hours.copy(),
                          self.coords.copy(),
                          self.locs.copy(),
                          self.quadkeys.copy(),
                          sequence_length,
                          batch_size,
                          split,
                          usage,
                          len(self.poi2id),
                          custom_seq_count)

    def user_count(self):
        return len(self.users)

    def locations(self):
        return len(self.poi2id)
    
    def quadkey_count(self):
        return self.n_quadkey

    def quadkey_location(self):
        return (self.quadkey_location_csr, self.quadkey_level_csr)
    
    def time_count(self):
        return [2, 7, 7*4, 24*7]   #6个小时为一个时间段
 
    def checkins_count(self):
        count = 0
        for loc in self.locs:
            count += len(loc)
        return count

    def read(self, file):
        if not os.path.isfile(file):
            print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(file))
            sys.exit(1)

        # collect all users with min checkins:
        self.read_users(file)
        # collect checkins for all collected users:
        self.read_pois(file)

    def read_users(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                # else:
                #    print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user
                visit_cnt = 1
                if 0 < self.max_users <= len(self.user2id):
                    break  # restrict to max users

    def read_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        # store location ids
        user_time = []
        user_coord = []
        user_loc = []
        user_weekday = []
        user_day = []
        user_time_slot = []
        user_hours = []
        user_quadkey = []
        count_num = [0] * len(self.map_level)
        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)  # from 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue  # user is not of interest(inactive user)
            user = self.user2id.get(user)  # from 0
            # if user == 3:
            #     break

            time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1,
                                                                                  1)).total_seconds()  # unix seconds
            # 自己加的time slot, 将一周的时间分成24 * 7个时间槽
            time_day = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")).weekday()
            hour_of_day = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")).hour
            time_hour = time_day * 24 + hour_of_day
            time_slot = time_day * 24/6 + math.floor(hour_of_day/6)
            if time_day == 0 or time_day == 6:
                time_week = 0
            else:
                time_week = 1
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)

            location = int(tokens[4])  # location nr
            if self.poi2id.get(location) is None:  # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
                self.poi2gps[self.poi2id[location]] = coord
            location = self.poi2id.get(location)  # from 0

            quadkey = []
            for index, m in enumerate(self.map_level):
                
                #quadkey_str = latlng2quadkey(lat, long, m) #+ '0' * (self.map_level[-1] - m)
                quadkey_str = latlng2quadkey(self.poi2gps[location][0], self.poi2gps[location][1], m)
                region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey_str, m)])
                #region_quadkey_bigram = region_quadkey_bigram.split()
                quadkey_int = int(quadkey_str)
                if quadkey_int not in self.quadkey2idx[index]:
                    self.quadkey2idx[index][quadkey_int] = self.n_quadkey[index]
                    self.idx2quadkey[index][self.n_quadkey[index]] = quadkey_int
                    self.n_quadkey[index] = self.n_quadkey[index] +  1
                quadkey_idx = self.quadkey2idx[index][quadkey_int]
                if quadkey_idx not in self.quadidx2loc[index]:
                    self.quadidx2loc[index][quadkey_idx] = [location]
                    count_num[index] += 1
                elif location not in self.quadidx2loc[index][quadkey_idx]:
                #else:
                    self.quadidx2loc[index][quadkey_idx].append(location)
                    count_num[index] += 1
                if index > 0:
                    if quadkey[-1] not in self.quadidx2level[index-1]:
                        self.quadidx2level[index-1][quadkey[-1]] = [quadkey_idx]
                    elif quadkey_idx not in self.quadidx2level[index-1][quadkey[-1]]:
                    #else:
                        self.quadidx2level[index-1][quadkey[-1]].append(quadkey_idx)
                quadkey.append(quadkey_idx)

                # if index != self.map_len-1:
                #     if quadkey[-1] not in self.quadkey_dict:
                #         self.quadkey_dict[quadkey[-1]] = []
                # if quadkey[-1] not in self.quadkey_dict[quadkey[-2]]:
                #     self.quadkey_dict[quadkey[-2]].append(quadkey[-1])


            if user == prev_user:
                # Because the check-ins for every user is sorted in descending chronological order in the file
                user_weekday.insert(0, time_week)
                user_day.insert(0, time_day)
                user_hours.insert(0, time_hour)
                user_time.insert(0, time)  # insert in front!
                user_time_slot.insert(0, time_slot)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
                user_quadkey.insert(0, quadkey)
            else:
                self.users.append(prev_user)  # 添加用户
                self.times.append(user_time)  # 添加列表
                self.time_slots.append(user_time_slot)
                self.time_weeks.append(user_weekday)
                self.time_days.append(user_day)
                self.time_hours.append(user_hours)
                self.coords.append(user_coord)
                self.locs.append(user_loc)
                self.quadkeys.append(user_quadkey)
                # print(len(user_time) == len(user_time_slot) == len(user_loc) == len(user_coord))
                # restart:
                prev_user = user
                user_time = [time]
                user_time_slot = [time_slot]
                user_weekday = [time_week]
                user_day = [time_day]
                user_hours = [time_hour]
                user_coord = [coord]
                user_loc = [location]
                user_quadkey = [quadkey]

        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.times.append(user_time)
        self.time_slots.append(user_time_slot)
        self.time_weeks.append(user_weekday)
        self.time_days.append(user_day)
        self.time_hours.append(user_hours)
        self.coords.append(user_coord)
        self.locs.append(user_loc)
        self.quadkeys.append(user_quadkey)


        #self.new_quadkeys2idx = {key: index for index, key in enumerate(sorted(self.quadkey2idx.keys()))}  #self.quadkey2idx[key]

        # self.loc2quadkey_bigram = ['NULL']
        # for l in range(len(self.poi2id)):
        #     lat, lng = self.poi2gps[l]
        #     quadkey = latlng2quadkey(lat, lng, self.map_level)
        #     quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, self.map_level)])
        #     quadkey_bigram = int(quadkey_bigram)    #.split()
        #     self.loc2quadkey_bigram.append(quadkey_bigram)
        #     self.all_quadkeys.append(quadkey_bigram)

        # self.QUADKEY = Field(
        #     sequential=True,
        #     use_vocab=True,
        #     batch_first=True,
        #     unk_token=None,
        #     preprocessing=str.split
        # )
        # self.QUADKEY.build_vocab(self.all_quadkeys)

        # for quadkey in self.quadkeys:
        #     user_quadkeys_ = []
        #     for user_q in quadkey:
        #         src_quadkeys_ = []
        #         for q in user_q:
        #             q_ = self.new_quadkeys2idx[q]
        #             src_quadkeys_.append(q_)
        #         user_quadkeys_.append(src_quadkeys_)
        #     self.quadkeys_num.append(user_quadkeys_)

        #print(self.new_quadkeys2idx)
        # print(len(self.quadkey2idx))
        # print("len", len(self.new_quadkeys2idx))
        # self.plt_fig()
    def create_csr_matrix(self):  #quadidx2level  quadidx2loc
        loc_matrix = []
        level_matrix = []
        for l in range(len(self.map_level)):
            rows = []
            cols = []
            data = []
            count_loc = 0
            for quad, locs in self.quadidx2loc[l].items():
                for loc in locs:
                    rows.append(quad)
                    cols.append(loc)
                    data.append(1)
                    count_loc += 1

            num_rows = max(rows) + 1
            num_cols = self.locations() + 1
            sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
            loc_matrix.append(sparse_matrix)
            if l != len(self.map_level)-1:
                rows = []
                cols = []
                data = []
                for low, highs in self.quadidx2level[l].items():
                    for high in highs:
                        rows.append(low)
                        cols.append(high)
                        data.append(1)
                num_rows = max(rows) + 1
                num_cols = self.n_quadkey[l+1]
                sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
                level_matrix.append(sparse_matrix)
        loc_matrix = {f'matrix_{i}': loc_matrix[i] for i in range(len(loc_matrix))}
        level_matrix = {f'matrix_{i}': level_matrix[i] for i in range(len(level_matrix))}
        np.savez('quadkey_location.npz', **loc_matrix)
        np.savez('quadkey_level.npz', **level_matrix)

        loaded_data = np.load('quadkey_location.npz', allow_pickle=True)
        # 提取稀疏矩阵
        quadkey_location_csr = [loaded_data[key].item() for key in loaded_data.files]
        self.quadkey_location_csr = []
        for sparse_matrix in quadkey_location_csr:
            coo = sparse_matrix.tocoo()
            indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
            values = torch.tensor(coo.data, dtype=torch.float32)
            shape = torch.Size(coo.shape)
            sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
            self.quadkey_location_csr.append(sparse_tensor)
            #print("sparse_tensor_sum:", sparse_tensor.to_dense().sum())


        loaded_data = np.load('quadkey_level.npz', allow_pickle=True)
        # 提取稀疏矩阵
        quadkey_level_csr = [loaded_data[key].item() for key in loaded_data.files]
        self.quadkey_level_csr = []
        for sparse_matrix in quadkey_level_csr:
            coo = sparse_matrix.tocoo()
            indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
            values = torch.tensor(coo.data, dtype=torch.float32)
            shape = torch.Size(coo.shape)
            sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
            self.quadkey_level_csr.append(sparse_tensor)

    def sparse_to_torch_tensor(self, sparse_matrix):
    #将 scipy.sparse.csr_matrix 转换为 PyTorch 稀疏张量
        # 获取稀疏矩阵的行、列和数据
        indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64)
        values = sparse_matrix.data.astype(np.float32)
        size = sparse_matrix.shape
        
        # 转换为 PyTorch 张量
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        size_tensor = torch.Size(size)
        
        # 创建 PyTorch 稀疏张量
        sparse_tensor = torch.sparse_coo_tensor(indices_tensor, values_tensor, size_tensor)
        # 转置稀疏张量和密集张量
        sparse_tensor_t = torch.sparse_coo_tensor(
        torch.stack([indices_tensor[1], indices_tensor[0]]),  # 交换行列索引
        values_tensor,
        torch.Size([size_tensor[1], size_tensor[0]])
        )
        return sparse_tensor_t

    def plt_fig(self):
        for user, user_seq in enumerate(self.locs):
            true = np.array(user_seq).flatten().tolist()
            plt.figure()
            # plt.plot(pred,color = 'red',label = 'Prediction')
            plt.plot(true,color = 'grey',label = 'Truth')
            plt.legend(loc="upper right")
            plt.title('Test Target')
            plt.savefig('./figure/train-{}.png'.format(user))
            plt.show()

        # for user, user_seq in enumerate(self.quadkeys_num):
        #     for k in range(len(self.map_level)):
        #         true = np.array(user_seq)[:,k].flatten().tolist()
        #         plt.figure()
        #         # plt.plot(pred,color = 'red',label = 'Prediction')
        #         plt.plot(true,color = 'grey',label = 'Truth')
        #         plt.legend(loc="upper right")
        #         plt.title('Test Target')
        #         plt.savefig('./figure/train-{}-{}.png'.format(user, k))
        #         plt.show()

