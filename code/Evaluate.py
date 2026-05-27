import matplotlib.pyplot as plt
import pandas as pd
import zstandard as zstd
from Environment import Environment
from Agent import Agent
import torch
import os

# TODO: config file
# MAIN
FILENAME = '../data/XNAS-20260524-6NCFWMPDHH/xnas-itch-20240523-20260522.mbo.csv.zst'

# ENVIRONMENT
TARGET_ORDER_SIZE = 100
TIME_WINDOW = pd.Timedelta(hours=1)
TRANSACTION_COST = 0.001

# AGENT
LEARNING_RATE = 0.0003
GAMMA = 0.99
EPSILON = 0.2
NUMBER_OF_EPISODES = 100

# ALL ORDER BOOK DATA
ALL_ORDER_BOOKS_FILENAME = '../data/all_order_books_data.pkl'
def decompress(file_name, num_data_points):
    dctx = zstd.ZstdDecompressor()
    with open(file_name, 'rb') as compressed_file:
        with dctx.stream_reader(compressed_file) as reader:
            df = pd.read_csv(reader, nrows=num_data_points)
    return df

if __name__ == "__main__":
    data = decompress(FILENAME,10_000)
    print(data.columns)
    print(pd.to_datetime(data['ts_event']))

    agent_environment = Environment(data, TARGET_ORDER_SIZE, TIME_WINDOW, TRANSACTION_COST)
    print("done with agent environment")
    agent = Agent(agent_environment, LEARNING_RATE, GAMMA, EPSILON, NUMBER_OF_EPISODES)
    print(agent.device)
    agent.load_state_dict(torch.load('../models/trained_agent.pth'))
    agent.eval()
    twap_environment = Environment(data, TARGET_ORDER_SIZE, TIME_WINDOW, TRANSACTION_COST)
    print("done with twap_environment")
    twap_metric_data = []
    agent_metric_data = []
    count = 0
    for episode in range(NUMBER_OF_EPISODES):
        valid_start_date = agent_environment.get_random_valid_date()
        agent_environment.reset(valid_start_date)
        twap_environment.reset(valid_start_date)
        
        state = agent_environment.get_state()
        is_done = False
        total_reward = 0
        num_of_steps = 0
        episode_state = {}
        while not is_done:
            action, _ = agent.select_action(state)
            state, reward, is_done = agent_environment.step(action)
            total_reward += reward
            num_of_steps += 1
        episode_state['total_reward'] = total_reward
        episode_state['num_of_steps'] = num_of_steps
        episode_state['total_slippage'] = agent_environment.get_average_execution_price() - agent_environment.get_start_price()
        episode_state['is_order_completed'] = agent_environment.get_is_order_completed()
        agent_metric_data.append(episode_state)

        is_done = False
        total_reward = 0
        num_of_steps = 0
        episode_state = {}
        while not is_done:
            action = TARGET_ORDER_SIZE / len(twap_environment.get_episode_data())
            _, reward, is_done = twap_environment.step(action)
            total_reward += reward
            num_of_steps += 1
        episode_state['total_reward'] = total_reward
        episode_state['num_of_steps'] = num_of_steps
        episode_state['total_slippage'] = twap_environment.get_average_execution_price() - twap_environment.get_start_price()
        episode_state['is_order_completed'] = twap_environment.get_is_order_completed()
        twap_metric_data.append(episode_state)
        count += 1
        print(f"{count} number of episodes have been completed!")

    agent_df = pd.DataFrame(agent_metric_data)
    twap_df = pd.DataFrame(twap_metric_data)

    fig, axes = plt.subplots(2,2)
    fig.suptitle("Agent vs TWAP")
    axes[0][0].plot(agent_df.index, agent_df['total_reward'], label = 'agent', color = 'blue', marker = 'o')
    axes[0][0].plot(twap_df.index, twap_df['total_reward'], label = 'twap', color = 'red', marker = 'o')
    axes[0][0].set_title("Agent vs TWAP: Total Reward")
    axes[0][0].set_xlabel("Episode number")
    axes[0][0].set_ylabel("Total Reward")
    axes[0][0].legend()

    axes[0][1].plot(agent_df.index, agent_df['num_of_steps'], label = 'agent', color = 'blue', marker = 'o')
    axes[0][1].plot(twap_df.index, twap_df['num_of_steps'], label = 'twap', color = 'red', marker = 'o')
    axes[0][1].set_title("Agent vs TWAP: Number of steps")
    axes[0][1].set_xlabel("Episode number")
    axes[0][1].set_ylabel("Number of steps")
    axes[0][1].legend()

    axes[1][0].plot(agent_df.index, agent_df['total_slippage'], label = 'agent', color = 'blue', marker = 'o')
    axes[1][0].plot(twap_df.index, twap_df['total_slippage'], label = 'twap', color = 'red', marker = 'o')
    axes[1][0].set_title("Agent vs TWAP: Total Slippage")
    axes[1][0].set_xlabel("Episode number")
    axes[1][0].set_ylabel("Total Slippage")
    axes[1][0].legend()

    axes[1][1].bar(['Agent', 'TWAP'], [agent_df['is_order_completed'].sum(), twap_df['is_order_completed'].sum()], color = 'blue')
    axes[1][1].set_title("Agent vs TWAP: Orders completed")
    axes[1][1].set_xlabel("Algorithm")
    axes[1][1].set_ylabel("Orders completed")

    fig.tight_layout()
    plt.show()

