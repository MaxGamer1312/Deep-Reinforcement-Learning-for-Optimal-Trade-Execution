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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        result = self.hidden_layers(state)
        result_actor = self.policy_head(result)
        result_critic = self.value_head(result)
        return (result_actor, result_critic)
    
    def select_action(self, state):
        state_tensor = torch.tensor(list(state.values()), dtype=torch.float32).to(self.device)
        result_actor, _ = self.forward(state_tensor)
        mean = result_actor[0]
        std = nn.functional.softplus(result_actor[1]) + 1e-6 # avoid zero
        normal_distribution = torch.distributions.Normal(mean, std)
        action = normal_distribution.sample()
        log_prob_action = normal_distribution.log_prob(action)
        return (action, log_prob_action)
    
    def store_experience(self, state, action, reward, log_prob_action, predicted_reward):
        self.experience_buffer.append((state, action, reward, log_prob_action, predicted_reward))
    
    def compute_advantages(self):
        advantage_list = []
        for _, _, reward, _, predicted_reward in self.experience_buffer:
            advantage = (reward - predicted_reward).detach()
            advantage_list.append(advantage)
        return advantage_list
    
    def update(self):
        advantage_list = self.compute_advantages()
        self.optimizer.zero_grad()
        total_loss = torch.zeros(1).to(self.device)
        for i in range(len(self.experience_buffer)):
            advantage = advantage_list[i]
            state, action, reward, prev_log_prob_action, predicted_reward = self.experience_buffer[i]
            result_actor, _ = self.forward(state)
            mean = result_actor[0]
            std = nn.functional.softplus(result_actor[1]) + 1e-6 # avoid zero
            normal_distribution = torch.distributions.Normal(mean, std)
            curr_log_prob_action = normal_distribution.log_prob(action)
            ratio = torch.exp(curr_log_prob_action - prev_log_prob_action)
            clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            actor_loss = -torch.min(ratio * advantage, clip_ratio * advantage)
            predicted_reward = predicted_reward.squeeze()
            critic_loss = nn.functional.mse_loss(predicted_reward, reward)
            total_loss = total_loss + actor_loss + critic_loss
        total_loss.backward()
        self.optimizer.step()
        self.experience_buffer = []

    def train(self):
        for _ in range(self.number_of_episodes):
            is_done = False
            self.environment.reset()
            state = self.environment.get_state()
            while not is_done:
                action, log_prob_action = self.select_action(state)
                state, reward, is_done = self.environment.step(action)
                state_tensor = torch.tensor(list(state.values()), dtype=torch.float32).to(self.device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
                _, result_critic = self.forward(state_tensor)
                self.store_experience(state_tensor, action, reward_tensor, log_prob_action, result_critic)
            self.update()

    