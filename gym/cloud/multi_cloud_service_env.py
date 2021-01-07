import numpy as np
import copy
import gym
import os
from gym import error, spaces, utils
from gym.utils import seeding
'''
    STATE:
    0. remain capacity in edge
    1. released VMs
    2. task size
    3. task length
    4. current spot instance service price
    5. reserved remain time
'''

class Constants:
    # 边缘节点参数设置
    EDGE_BASIC_COST = 0.03
    EDGE_WORK_COST = 0.2065
    EDGE_COEFFICIENT = 1
    TOTAL_TASK_NUM = 100 #

    EDGE_CAPACITY = 90

    # 用户任务参数设置
    TASK_SIZE_MEAN = 10 #
    TASK_SIZE_STD = 5
    MIN_TASK_SIZE = 1
    MAX_TASK_SIZE = TASK_SIZE_MEAN + 3 * TASK_SIZE_STD

    TASK_LENGTH_MEAN = 10  #
    TASK_LENGTH_STD = 5
    MIN_TASK_LENGTH = 1
    MAX_TASK_LENGTH = TASK_LENGTH_MEAN + 3 * TASK_LENGTH_STD

    # 云服务价格及类型参数设置

    # On Demand Instance Pricing
    SERVICE_OD1_PRICE = 3.0

    # Spot Instance Pricing
    SERVICE_SI1_PRICE_MEAN = 1.5
    SERVICE_SI1_PRICE_STD = 1.0
    MIN_SERVICE_SI1_PRICE = 1.0
    MAX_SERVICE_SI1_PRICE = SERVICE_SI1_PRICE_MEAN + 3 * SERVICE_SI1_PRICE_STD

    # Reserved Instance Pricing
    SERVICE_RE1_UPFRONT = 800
    SERVICE_RE1_PERIOD = 20 #
    SERIVCE_RE1_PRICE = 1.5

    SHOW_STEP = False

    # 设置缩放向量
    SCALE_VECTOR = np.array([EDGE_CAPACITY, EDGE_CAPACITY, MAX_TASK_SIZE, MAX_TASK_LENGTH, MAX_SERVICE_SI1_PRICE, SERVICE_RE1_PERIOD])


class MCSEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=0, edge_capacity=Constants.EDGE_CAPACITY,
                 task_size_mean=Constants.TASK_SIZE_MEAN, task_size_std=Constants.TASK_SIZE_STD,
                 task_length_mean=Constants.TASK_LENGTH_MEAN, task_length_std=Constants.TASK_LENGTH_STD,
                 service_si1_price_mean=Constants.SERVICE_SI1_PRICE_MEAN, service_si1_price_std=Constants.SERVICE_SI1_PRICE_STD,
                 service_re1_upfront=Constants.SERVICE_RE1_UPFRONT, service_re1_period=Constants.SERVICE_RE1_PERIOD, service_re1_price=Constants.SERIVCE_RE1_PRICE,
                 service_od1_price=Constants.SERVICE_OD1_PRICE
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
        self.spot_usage_record = [] # spot instance 使用记录
        self.reserved_usage_record = [] # reserved 使用记录

        self.service_od1_price = service_od1_price

        self.service_re1_price = service_re1_price  # 预付费类型1 云服务单价
        self.service_re1_period = service_re1_period  # 预付费资源周期
        self.service_re1_upfront = service_re1_upfront  # 预付费价格
        self.service_re1_remain_time = 0  # 预付费类型1剩余时间
        self.service_re1_is_available = False  # 预付费类型1 是否可用

        self.service_si1_price_mean = service_si1_price_mean
        self.service_si1_price_std = service_si1_price_std

        num_actions = 3

        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),
            # spaces.Box(0.0, 1.0, shape=(2, ), dtype=np.float32)
            spaces.Tuple(
                tuple(spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
                for i in range(num_actions))
            )
        ))
        # Observation Space的结构是否还需要改动
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0., high=1., shape=(6, ), dtype=np.float32),
            spaces.Discrete(200),
        ))

        self.window = None

        self.remain_capacity = self.edge_capacity
        self.released_vm = 0

        self.cap_record = []

        # 使用量统计
        self.total_od = 0
        self.total_reserved = 0
        self.total_spot = 0

    def reset(self):
        np.random.seed(self.seed)
        self.task_counter = 0
        self.ep_r = 0
        self.ep_r_trace = []

        self.service_re1_is_available = False
        self.service_re1_remain_time = 0

        self.remain_capacity = self.edge_capacity

        self.usage_record = []  # 边缘节点的使用记录
        self.spot_usage_record = []  # spot instance 使用记录
        self.reserved_usage_record = []  # reserved 使用记录

        self.done = False

        self.released_vm = 0
        state = self.get_state().copy()
        return copy.deepcopy(state)

    def step(self, action):
            """
            :param action:
            action = (act_index, [param1, param2])
            两个部分 一部分是选择的动作的编号， act , 另外一部分是动作的参数列表，未选择的动作使用0填充
            act_index 0 租赁云服务器0-on demand上的资源进行分配，参数值代表分配到云服务器0上的资源，剩余资源由边缘节点提供
            act_index 1 租赁云服务器1-reserved类型上的资源进行分配，参数值代表分配到云服务器1上的资源，剩余资源由边缘节点提供
            act_index 2 租赁云服务器2-spot-instance 类型上的资源进行分配，参数值代表分配到服务器2上的资源，剩余资源由边缘节点提供
            :return: 执行后的状态，奖励值，停止信号等
            """

            cost = 0  # 成本
            edge_cost, cloud_cost = 0, 0

            act_index = action[0]  # 动作索引
            act_param = action[1][act_index][0]  # 所选动作参数
            act_param = np.clip(act_param, 0, 1) # 动作参数裁剪，避免加噪音

            # 获取服务器当前的状态
            state = copy.deepcopy(self.state)
            remain_capacity = state[0]
            task_size = state[2]
            task_length = state[3]
            spot_price = state[4]

            # 根据动作计算需要租赁的云服务器和边缘节点VM数量
            cloud_vm = int(np.floor(task_size * act_param))
            edge_vm = task_size - cloud_vm

            # 边缘服务器承载能力不足， 将边缘服务器装满后，剩余的资源由云服务器提供
            if edge_vm > remain_capacity:
                cloud_vm = task_size - remain_capacity
                edge_vm = remain_capacity

            on_demand_cost = 0
            reserved_cost = 0
            spot_cost = 0

            # 选择租赁 on demand类型云服务器
            if act_index == 0:
                # 计算需要的成本
                on_demand_cost = self.service_od1_price * cloud_vm * task_length  # 云服务器最终成本
                self.total_od += cloud_vm

            # 选择租赁 reserves类型云服务器
            elif act_index == 1:
                # 检测服务是否可用
                if not self.service_re1_is_available:
                    reserved_cost += self.service_re1_upfront
                    self.service_re1_remain_time = self.service_re1_period
                    self.service_re1_is_available = True
                self.reserved_usage_record.append([cloud_vm, task_length])

            # 选择租赁 spot instance 类型云服务器
            elif act_index == 2:
                self.spot_usage_record.append([cloud_vm, task_length])
            # 计算本时间槽内reserved类型服务器的维护费用
            total_reserved_vm = 0
            for req in self.reserved_usage_record:
                total_reserved_vm += req[0]

            reserved_cost += Constants.SERIVCE_RE1_PRICE * total_reserved_vm
            # 计算本时间槽内spot instance 类型服务器的租赁费用
            total_spot_instance = 0
            for req in self.spot_usage_record:
                total_spot_instance += req[0]
            spot_cost = spot_price * total_spot_instance

            # 计算本时间槽内的总费用
            cloud_cost = on_demand_cost + reserved_cost + spot_cost
            # 计算边缘节点的工作负载及其工作成本
            remain_capacity = remain_capacity - edge_vm
            # edge_workload = 1 - remain_capacity / self.edge_capacity
            # edge_cost_coefficient = Constants.EDGE_COEFFICIENT * (1 + edge_workload) * Constants.EDGE_BASIC_COST
            # edge_cost = np.round(edge_cost_coefficient * (self.edge_capacity - remain_capacity), 4) # 边缘节点的成本计算\
            edge_cost = np.round(remain_capacity * Constants.EDGE_BASIC_COST + (self.edge_capacity - remain_capacity) * Constants.EDGE_WORK_COST, 4)

            # 计算系统的总成本
            cost = edge_cost + cloud_cost

            # 更新系统状态
            self.usage_record.append([edge_vm, task_length])
            self.released_vm = self.update_record()
            self.update_spot_record()
            self.update_reserved_record()

            remain_capacity += self.released_vm
            self.remain_capacity = remain_capacity


            if self.service_re1_is_available:
                self.service_re1_remain_time -= 1

            if self.service_re1_remain_time == 0:
                self.service_re1_is_available = False

            self.task_counter += 1

            next_state = self.get_state().copy()

            # Show selected action
            if Constants.SHOW_STEP:
                print(f"STEP: {self.task_counter}, TYPE: {act_index}, PARAM: {act_param}, CAP: {self.remain_capacity}, Released: {self.released_vm}, Edge Cost: {edge_cost} Cloud Cost: {cloud_cost}, Req: {task_size, task_length}")

            info = {}
            # 记录剩余资源情况
            self.cap_record.append(self.remain_capacity)


            return next_state, -cost/500, self.done, info

    def update_record(self):
        delete_index = []
        released_vm = 0
        records = copy.deepcopy(self.usage_record)
        for i in range(len(records)):
            records[i][1] = records[i][1] - 1
            if records[i][1] == 0:
                released_vm += records[i][0]
                delete_index.append(i)

        records = [records[i] for i in range(len(records)) if (i not in delete_index)]
        self.usage_record = copy.deepcopy(records)
        return released_vm

    def update_reserved_record(self):
        delete_index = []
        released_vm = 0
        records = copy.deepcopy(self.reserved_usage_record)
        for i in range(len(records)):
            records[i][1] = records[i][1] - 1
            if records[i][1] == 0:
                released_vm += records[i][0]
                delete_index.append(i)

        records = [records[i] for i in range(len(records)) if (i not in delete_index)]
        self.reserved_usage_record = copy.deepcopy(records)
        return released_vm

    def update_spot_record(self):
        delete_index = []
        released_vm = 0
        records = copy.deepcopy(self.spot_usage_record)
        for i in range(len(records)):
            records[i][1] = records[i][1] - 1
            if records[i][1] == 0:
                released_vm += records[i][0]
                delete_index.append(i)

        records = [records[i] for i in range(len(records)) if (i not in delete_index)]
        self.spot_usage_record = copy.deepcopy(records)
        return released_vm

    def task_generator(self):
        # 生成任务数据, 四舍五入制
        while True:
            task_size = int(np.around(np.random.normal(self.task_size_mean, self.task_size_std), 1))
            if task_size >= 1 and task_size <= Constants.MAX_TASK_SIZE:
                break

        while True:
            task_length = int(np.around(np.random.normal(self.task_length_mean, self.task_length_std), 1))
            if task_length >= 1 and task_length <= Constants.MAX_TASK_LENGTH:
                break

        return [task_size, task_length]

    def cloud_service_price_generator(self):
        while True:
            price = np.around(np.random.normal(self.service_si1_price_mean, self.service_si1_price_std), 4)
            if price >= Constants.MIN_SERVICE_SI1_PRICE and price <= Constants.MAX_SERVICE_SI1_PRICE:
                return price

    def random_action(self):
        index = int(np.random.choice(3, 1))
        param = np.round(np.random.rand(), 4)
        all_params = [np.zeros((1,)), np.zeros((1,)), np.zeros((1,))]
        all_params[index][:] = param
        return (index, all_params)

    def _scale_state(self, state):
        scaled_state = state / Constants.SCALE_VECTOR
        return scaled_state

    def get_state(self):
        if self.task_counter == Constants.TOTAL_TASK_NUM:
            self.done = True
            self.state = [self.remain_capacity, self.released_vm, 0, 0, 0, 0]
        else:
            task_info = self.task_generator()
            cloud_price = self.cloud_service_price_generator()

            self.state = [self.remain_capacity, self.released_vm, task_info[0], task_info[1], cloud_price, self.service_re1_remain_time]

        state = self.state.copy()
        scaled_state = self._scale_state(state)
        return scaled_state

    def write_txt(self):
        pass


if __name__ == '__main__':
    env = MCSEnv(seed=1)
    done = False
    ep_r = 0
    ep_num = 0
    ep_steps = 0
    env.reset()

    # for episode in range(10):
    for t in range(1):
        while True:
            ep_steps += 1
            action = env.random_action()
            next_state, reward, done, _ = env.step(action)
            ep_r += reward
            if done:
                print(f"Episode Num: {ep_num + 1} Episode Steps: {ep_steps} Reward: {ep_r}")
                state, done = env.reset(), False
                ep_r = 0
                ep_steps = 0
                ep_num += 1
                env.reset()
                break


