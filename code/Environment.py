import pandas as pd
import random
from OrderBook import OrderBook
import copy

class Environment():
    def __init__(self, data, target_order_size, time_window, transaction_costs, valid_date = None):
        self.data = data
        self.target_order_size = target_order_size
        self.time_window = time_window
        self.transaction_costs = transaction_costs
        self.temp_order_book = OrderBook()
        self.data['ts_event'] = pd.to_datetime(self.data['ts_event'])
        self.reset(valid_date)

    def reset(self, valid_date = None):
        if valid_date:
            random_start_date = valid_date
        else:
            random_start_date = self.get_random_valid_date()
        self.shares_bought = 0
        self.current_cost = 0
        self.start_market_event = self.data[self.data['ts_event'] == random_start_date].sort_values(by='ts_event').iloc[0]
        self.current_market_event = self.start_market_event
        self.end_time = self.start_market_event['ts_event'] + self.time_window
        self.start_price = float(self.current_market_event['price'])
        self.current_market_event_index = 0
        self.episode_data = self.data[(self.data['ts_event'] >= random_start_date) & (self.data['ts_event'] <= self.end_time)].sort_values(by='ts_event')
        self.order_book = OrderBook()
        for market_event in self.data[self.data['ts_event'] < random_start_date].to_dict('records'):
            self.order_book.update(market_event)
        self.welford_calculations = {}
        self.welford_calculations['count'] = 0
        self.welford_calculations['mean'] = 0.0
        self.welford_calculations['M2'] = 0.0

    def step(self, shares_wanted):
        shares_wanted = float(shares_wanted)
        if(shares_wanted < 0):
            shares_wanted = 0
        if(shares_wanted > (self.target_order_size - self.shares_bought)):
            shares_wanted = self.target_order_size - self.shares_bought
        current_shares_left = shares_wanted
        current_share_ask_index = 0
        current_share_ask_list = self.order_book.get_asks()
        while(current_shares_left != 0 and self.current_market_event_index < len(self.episode_data)):
            while(not current_share_ask_list or current_share_ask_index >= len(current_share_ask_list)):
                self.current_market_event_index += 1
                if(self.current_market_event_index >= len(self.episode_data)):
                    break
                self.current_market_event = self.episode_data.iloc[self.current_market_event_index]
                current_share_ask_index = 0
                self.order_book.update(self.current_market_event)
                current_share_ask_list = self.order_book.get_asks()
            if(self.current_market_event_index >= len(self.episode_data)):
                break
            current_share_ask = current_share_ask_list[current_share_ask_index]
            prev_shares_left = current_shares_left
            current_shares_left = float(max(0, current_shares_left - current_share_ask['size']))
            self.order_book.consume(current_share_ask['order_id'], prev_shares_left - current_shares_left)
            self.current_cost += float(current_share_ask['price'] * (prev_shares_left - current_shares_left))
            current_share_ask_index += 1
        # if didn't happen to buy all
        self.shares_bought += shares_wanted - current_shares_left
        self.current_cost += self.transaction_costs
        return (self.get_state(), self._get_reward(), self.shares_bought == self.target_order_size or self.current_market_event_index >= len(self.episode_data))

    def _get_reward(self):
        if(self.shares_bought == 0):
            return 0
        if(self.shares_bought != self.target_order_size and self.current_market_event_index >= len(self.episode_data)):
            return -1
        return (self.start_price - (self.current_cost / self.shares_bought)) / self.start_price

    def get_state(self):
        state = {}
        #TODO: normalize features?
        state['current_price'] = float(self.current_market_event['price'] / 1000)
        state['shares_remaining_to_buy'] = (self.target_order_size - self.shares_bought) / self.target_order_size
        state['shares_bought'] = self.shares_bought / self.target_order_size
        state['time_remaining'] = float((self.end_time - self.current_market_event['ts_event']).total_seconds() / pd.Timedelta(self.time_window).total_seconds())
        if(self.shares_bought == 0):
            state['average_execution_price'] = 0
        else:
            state['average_execution_price'] = float((self.current_cost / self.shares_bought) / 1000)
        self.welford_calculations['count'] += 1
        delta = float(self.current_market_event['price'] - self.welford_calculations['mean'])
        self.welford_calculations['mean'] += delta / self.welford_calculations['count']
        delta2 = float(self.current_market_event['price'] - self.welford_calculations['mean'])
        self.welford_calculations['M2'] += delta * delta2
        if(self.welford_calculations['count'] < 2):
            state['current_market_volatility'] = 0
        else:
            # Using Welford's algorithm for efficiency
            state['current_market_volatility'] = float((self.welford_calculations['M2'] / self.welford_calculations['count']) ** 0.5 / 10)
        return state
    def get_episode_data(self):
        return self.episode_data
    
    def get_average_execution_price(self):
        if self.shares_bought == 0:
            return 0
        return self.current_cost / self.shares_bought
    def get_start_price(self):
        return self.start_price
    def get_is_order_completed(self):
        # TODO: can you buy more than the target order size?
        return self.shares_bought == self.target_order_size
    
    def get_random_valid_date(self):
        
        valid_dates = self.data[self.data['ts_event'] + self.time_window <= self.data['ts_event'].max()]['ts_event'].unique()
        random_date = random.choice(valid_dates)
        return random_date
        
