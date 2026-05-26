import pandas as pd
import zstandard as zstd
from Environment import Environment
from Agent import Agent
import torch

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
    agent = Agent(agent_environment, LEARNING_RATE, GAMMA, EPSILON, NUMBER_OF_EPISODES)
    print(agent.device)
    agent.load_state_dict(torch.load('../models/trained_agent.pth'))

    twap_environment = Environment(data, TARGET_ORDER_SIZE, TIME_WINDOW, TRANSACTION_COST)
    
    twap_metric_data = []
    agent_metric_data = []
    for episode in range(NUMBER_OF_EPISODES):
        valid_start_date = agent_environment.get_random_valid_date()
        agent_environment.reset(valid_start_date)
        twap_environment.reset(valid_start_date)
        
        is_done = False
        total_reward = 0
        total_slippage = 0
        is_order_completed = True
        num_of_steps = 0
        while not is_done:
            action, log_prob_action = agent.select_action(state)
            state, reward, is_done = agent_environment.step(action)

        is_done = False
        while not is_done:


    
        