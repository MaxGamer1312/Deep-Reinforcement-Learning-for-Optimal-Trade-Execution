import torch.nn as nn
import torch

class Agent(nn.Module):
    def __init__(self, environment, learning_rate, gamma, epsilon, number_of_episodes):
        super().__init__()
        self.environment = environment
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.number_of_episodes = number_of_episodes
        self.hidden_layers = nn.Sequential(
            nn.Linear(len(self.environment.get_state().keys()),64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, 2)
        self.value_head = nn.Linear(64, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.experience_buffer = []

    def forward(self, state):
        result = self.hidden_layers(state)
        result_actor = self.policy_head(result)
        result_critic = self.value_head(result)
        return (result_actor, result_critic)
    
    def select_action(self, state):
        state_tensor = torch.tensor(list(state.values()), dtype=torch.float32)
        result_actor, _ = self.forward(state_tensor)
        mean = result_actor[0]
        std = nn.functional.softplus(result_actor[1])
        normal_distribution = torch.distributions.Normal(mean, std)
        action = normal_distribution.sample()
        log_prob_action = normal_distribution.log_prob(action)
        return (action, log_prob_action)
    
    def store_experience(self, state, action, reward, log_prob_action, predicted_reward):
        self.experience_buffer.append((state, action, reward, log_prob_action, predicted_reward))
    
    def compute_advantages(self):
        advantage_list = []
        for _, _, reward, _, predicted_reward in self.experience_buffer:
            advantage = reward - predicted_reward
            advantage_list.append(advantage)
        return advantage_list
    
    def update(self):
        advantage_list = self.compute_advantages()
        for i in range(len(self.experience_buffer)):
            advantage = advantage_list[i]
            state, action, reward, prev_log_prob_action, predicted_reward = self.experience_buffer[i]
            state_tensor = torch.tensor(list(state.values()), dtype=torch.float32)
            result_actor, _ = self.forward(state_tensor)
            mean = result_actor[0]
            std = nn.functional.softplus(result_actor[1])
            normal_distribution = torch.distributions.Normal(mean, std)
            curr_log_prob_action = normal_distribution.log_prob(action)
            ratio = torch.exp(curr_log_prob_action - prev_log_prob_action)
            clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            actor_loss = -torch.min(ratio * advantage, clip_ratio * advantage)
            critic_loss = nn.functional.mse_loss(predicted_reward, reward)
            total_loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        self.experience_buffer = []

    def train(self):
        

    