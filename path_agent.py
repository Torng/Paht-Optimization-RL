import config
from model_logger import gantt_result, clear_result
import torch
import random
import math
from Module.dueling_dqn import DuelingDQN
from model.replay_memory import ReplayMemory, Transition
import torch.nn as nn
from schedule_kpi import ScheduleKPI


class PathAgent:
    def __init__(self, policy_net, device, job_count, attribute_count, action_count, optimizer):
        self.device = device
        self.policy_net = policy_net
        self.target_net = DuelingDQN(job_count, attribute_count, action_count, device).to(device)
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(config.REPLAY_BUFFER_CAPACITY)
        self.job_count = job_count
        self.attribute_count = attribute_count
        self.action_count = action_count
        self.optimizer = optimizer
        self.steps_done = 0
        self.best_observation = None
        self.best_kpi_score = 0

    def test_agent(self, env, job_count: int, equipment_list, display_status: bool = True, num_episodes=5):
        for i in range(num_episodes):
            state = env.reset()
            if display_status:
                print("num_episodes--->", i)
            step = 0
            done = False
            total_return = 0
            episode_return = 0
            if display_status:
                while not done:
                    print("step {}:".format(step))
                    print(gantt_result(env.job_info, equipment_list))
                    # print(clear_result(state, job_count))
                    action = self.select_action(state)
                    state, reward, done = env.step(action.item())
                    if done:
                        clear_result(state, job_count, len(equipment_list))
                    step += 1
            else:
                while not done:
                    action = self.select_action(state)
                    state, reward, done = env.step(action.item())
                    episode_return += reward
                total_return += episode_return
        avg_return = total_return / num_episodes
        return avg_return

    def select_action(self, state):
        sample = random.random()
        eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * \
                        math.exp(-1. * self.steps_done / config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(-1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_count)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        transitions = self.memory.sample(config.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(config.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * config.DISCOUNT) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_best_observation(self, test_env, equipments_list, job_info_list):
        state = test_env.reset()
        done = False
        best_observation_step = []
        while not done:
            best_observation_step.append(state.numpy())
            action = self.select_action(state)
            state, reward, done = test_env.step(action.item())
        self.best_observation = best_observation_step
