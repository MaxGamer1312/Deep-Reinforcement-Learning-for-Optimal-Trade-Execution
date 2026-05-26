import pandas as pd
import zstandard as zstd
from Environment import Environment
from Agent import Agent
# TODO: config file
# MAIN
FILENAME = '../data/XNAS-20260524-6NCFWMPDHH/xnas-itch-20240523-20260522.mbo.csv.zst'

# ENVIRONMENT
TARGET_ORDER_SIZE = 10_000
TIME_WINDOW = pd.Timedelta(days=7)
TRANSACTION_COST = 0.01

# AGENT
LEARNING_RATE = 0.0003
GAMMA = 0.99
EPSILON = 0.2
NUMBER_OF_EPISODES = 1000


def decompress(file_name, num_data_points):
    dctx = zstd.ZstdDecompressor()
    with open(file_name, 'rb') as compressed_file:
        with dctx.stream_reader(compressed_file) as reader:
            df = pd.read_csv(reader, nrows=num_data_points)
    return df

if __name__ == "__main__":
    data = decompress(FILENAME,1_000_000)
    environment = Environment(data, TARGET_ORDER_SIZE, TIME_WINDOW, TRANSACTION_COST)
    agent = Agent(environment, LEARNING_RATE, GAMMA, EPSILON, NUMBER_OF_EPISODES)