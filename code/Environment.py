import pandas as pd
import random
from OrderBook import OrderBook

class Environment():
    def __init__(self, data, target_order_size, time_window, transaction_costs):
        self.data = data
        self.target_order_size = target_order_size
        self.time_window = time_window
        self.transaction_costs = transaction_costs
        self.reset()

    def reset(self):
        random_start_date = self._get_random_valid_date()
        self.shares_bought = 0
        self.current_cost = 0
        self.start_market_event = self.data[self.data['ts_event'] == random_start_date].sort_values(by='ts_event').iloc[0]
        self.current_market_event = self.start_market_event
        self.end_time = self.start_market_event['ts_event'] + self.time_window
        self.start_price = self.current_market_event['price']
        self.current_market_event_index = 0
        self.episode_data = self.data[(self.data['ts_event'] >= random_start_date) & (self.data['ts_event'] <= self.end_time)].sort_values(by='ts_event')
        self.order_book = OrderBook()
        all_market_events_before_and_including_start = self.data[self.data['ts_event'] <= random_start_date].sort_values(by='ts_event')
        for _, market_event in all_market_events_before_and_including_start.iterrows():
            self.order_book.update(market_event)

    def step(self, shares_wanted):
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
            current_shares_left = max(0, current_shares_left - current_share_ask['size'])
            self.order_book.consume(current_share_ask['order_id'], prev_shares_left - current_shares_left)
            self.current_cost += current_share_ask['price'] * (prev_shares_left - current_shares_left)
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
        state['current_price'] = self.current_market_event['price']
        state['shares_remaining_to_buy'] = self.target_order_size - self.shares_bought
        state['shares_bought'] = self.shares_bought
        state['time_remaining'] = (self.end_time - self.current_market_event['ts_event']).total_seconds()
        if(self.shares_bought == 0):
            state['average_execution_price'] = 0
        else:
            state['average_execution_price'] = self.current_cost / self.shares_bought
        current_market_price_list = self.episode_data[(self.episode_data['ts_event'] >= self.start_market_event['ts_event']) & (self.episode_data['ts_event'] <= self.current_market_event['ts_event'])]['price']
        if(len(current_market_price_list) < 2):
            state['current_market_volatility'] = 0
        else:
            state['current_market_volatility'] = current_market_price_list.std()
        return state
    
    def _get_random_valid_date(self):
        self.data['ts_event'] = pd.to_datetime(self.data['ts_event'])
        valid_dates = self.data[self.data['ts_event'] + self.time_window <= self.data['ts_event'].max()]['ts_event'].unique()
        random_date = random.choice(valid_dates)
        return random_date
        
