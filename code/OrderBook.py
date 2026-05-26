
class OrderBook():
    def __init__(self):
        self.active_orders = {}
    
    def update(self, market_event):
         match market_event['action']:
             case 'A':
                 self.active_orders[market_event['order_id']] = market_event
             case 'R' | 'T' | 'F' | 'C':
                 if market_event['order_id'] in self.active_orders:
                     del self.active_orders[market_event['order_id']]
    def consume(self, order_id, shares_consumed):
        if order_id in self.active_orders:
            if self.active_orders[order_id]['size'] <= shares_consumed:
                del self.active_orders[order_id]
            else:
                self.active_orders[order_id]['size'] -= shares_consumed
    def get_asks(self):
        return sorted(list(ask for ask in self.active_orders.values() if ask['side'] == 'A'), key=lambda x:x['price'])
    def get_bids(self):
        return sorted(list(bid for bid in self.active_orders.values() if bid['side'] == 'B'), key=lambda x:x['price'], reverse=True)