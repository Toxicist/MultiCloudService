import numpy as np
import copy
import random


class MCSEnv(object):
    def __init__(self, seed=0, edge_capacity=600,
                 task_size_mean=10, task_size_std=3,
                 task_length_mean=5, task_length_std=2,
                 service_od1_price_mean=5, service_od1_price_std=1,
                 service_od2_price_mean=5, service_od2_price_std=1,
                 service_re1_upfront=300, service_re1_period=30, service_re1_price=2,
                 service_re2_upfront=500, service_re2_peroid=30, service_re2_price=1.2
                 ):
        self.seed = seed
        np.random.seed(self.seed)

        self.edge_capacity = edge_capacity
        self.task_size_mean = task_size_mean
        self.task_size_std = task_size_std
        self.task_length_mean = task_length_mean
        self.task_length_std = task_length_std

        self.task_counter = 0
        self.ep_r = 0
        self.ep_r_trace = []

        self.done = False

        self.state = []

        self.usage_record = []  # 边缘节点的使用记录

        self.service_od1_price_std = service_od1_price_std
        self.service_od1_price_mean = service_od1_price_mean
        self.service_od2_price_std = service_od2_price_std
        self.service_od2_price_mean = service_od2_price_mean

        self.service_re1_price = service_re1_price
        self.service_re1_period = service_re1_period
        self.service_re1_upfront = service_re1_upfront

        self.service_re2_price = service_re2_price
        self.service_re2_peroid = service_re2_peroid
        self.service_re2_upfront = service_re2_upfront

    def reset(self):
        np.random.seed(self.seed)
        self.task_counter = 0
        self.ep_r = 0
        self.ep_r_trace = []

        self.remain_capacity = self.edge_capacity
        self.usage_record = []

        self.done = False
        '''
            STATE:
            1. remain capacity in edge
            2. released VMs
            3. task size
            4. task length
            5.
        '''
        task_info = self.task_generator()


    def step(self, action):
        pass

    def update_record(self):
        pass

    def task_generator(self):
        # 生成任务数据, 四舍五入制
        while True:
            task_size = int(np.around(np.random.normal(self.task_size_mean, self.task_size_std), 1))
            if task_size >= 1:
                break

        while True:
            task_length = int(np.around(np.random.normal(self.task_length_mean, self.task_length_std), 1))
            if task_length >= 1:
                break

        return [task_size, task_length]

    def service_price_generator(self):
        pass


if __name__ == '__main__':
    env = MCSEnv()
    print(env.task_generator())
