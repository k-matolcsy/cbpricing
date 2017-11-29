import datetime as dt
import numpy as np
from bsmarket import Option


class BinomialTree(object):
    def __init__(self, steps):
        self.size = steps + 1
        self.tree = [np.empty(x+1, dtype=tuple) for x in range(self.size)]

    def __str__(self):
        return str(self.tree)


class CRRTree(BinomialTree):

    def __init__(self, steps, ticker, option_type='call', maturity=1, strike=None):
        # pass input to parent
        super().__init__(steps)

        # save inputs
        self.steps = steps
        self.ticker = ticker
        self.option_type = option_type
        self.maturity = maturity        # years
        self.strike = strike

        # objects
        expiry = str(dt.datetime.today() + dt.timedelta(days=365*maturity))[:10]
        self.option = Option(ticker, option_type, expiry, strike)
        self.stock = self.option.stock
        self.yc = self.option.yield_curve

        # set parameters
        self.volatility = max(self.stock.vol_hist, self.option.vol_implied())
        print("""
            Volatility
            historical: {}
            implied: {} 
            """.format(self.stock.vol_hist, self.option.vol_implied()))
        self.dt = self.maturity / self.steps
        self.up = np.exp(self.volatility * self.dt ** 0.5)
        self.down = 1 / self.up

        # build stock tree
        self.__build_stock()

        # build option tree
        print(self.option.price)
        print(self.option.maturity)
        self.__build_derivative()

    def __build_stock(self):
        self.tree[0][0] = self.stock.price
        for i in range(1, self.size):
            for j in range(i, self.size):
                self.tree[j][i - 1] = self.tree[j - 1][i - 1] * self.up
            self.tree[i][i] = self.tree[i - 1][i - 1] * self.down
        return None

    def __payoff_function(self, stock):
        if self.option_type == 'call':
            return max(stock - self.strike, 0)
        elif self.option_type == 'put':
            return max(self.strike - stock, 0)
        else:
            print("Wrong option_type")
            raise Exception

    def __prob(self, step):
        f = self.yc.forward(step * self.dt, (step + 1) * self.dt)
        return ((1 + f) ** self.dt - self.down) / (self.up - self.down)

    def __build_derivative(self):
        for i in range(self.size):
            self.tree[-1][i] = (self.tree[-1][i], self.__payoff_function(self.tree[-1][i]))
        for j in reversed(range(self.size-1)):
            p = self.__prob(j)
            df = self.yc.df((j+1)*self.dt, j*self.dt)
            for i in range(j+1):
                self.tree[j][i] = (self.tree[j][i], df * (p * self.tree[j+1][i][1] + (1-p) * self.tree[j+1][i+1][1]))


class ConvertibleTree(BinomialTree):

    def __init__(self, principal, maturity, coupon, coupon_freq, steps, ticker, con_ratio):
        super().__init__(steps)

        # save inputs
        self.principal = principal
        self.maturity = maturity        # years
        self.coupon = coupon
        self.coupon_freq = coupon_freq
        self.steps = steps
        self.ticker = ticker
        self.cr = con_ratio
        # call provision
        self.call = 1100
        # dilution
        self.stock_out = 168.07     # million
        self.cb_out = 1.2  # million
        # continuous dividend
        self.div_cont = 0.00
        # discrete dividend
        self.div_disc = 0.00
        self.div_freq = 2

        # objects
        expiry = str(dt.datetime.today() + dt.timedelta(days=365*maturity))[:10]
        strike = self.principal / self.cr       # conversion price?
        self.option = Option(ticker, 'call', expiry, strike)
        self.stock = self.option.stock
        self.yc = self.option.yield_curve

        # set parameters
        self.volatility = max(self.stock.vol_hist, self.option.vol_implied())
        print("""
            Volatility
            historical: {}
            implied: {} 
            """.format(self.stock.vol_hist, self.option.vol_implied()))
        self.dt = self.maturity / self.steps
        self.up = np.exp(self.volatility * self.dt ** 0.5)
        self.down = 1 / self.up

        # build stock tree
        self.__build_stock()

        # build option tree
        print("option price: " + str(self.option.price))
        print("option maturity: " + str(self.option.maturity))
        self.__build_derivative()

    def __build_stock(self):
        self.tree[0][0] = self.stock.price
        for i in range(1, self.size):
            for j in range(i, self.size):
                self.tree[j][i - 1] = self.tree[j - 1][i - 1] * self.up * (1 - self.__dividend(j))
            self.tree[i][i] = self.tree[i - 1][i - 1] * self.down * (1 - self.__dividend(i))
        return None

    def __payoff_function(self, stock):
        stock_at_con = (self.stock_out * stock + self.cb_out * self.principal) / \
                       (self.stock_out + self.cb_out * self.cr)
        # conversion after or before coupon?
        return max(self.principal, self.cr * stock_at_con) + self.coupon * self.principal

    def __prob(self, step):
        f = self.yc.forward(step * self.dt, (step + 1) * self.dt)
        return ((1 + f - self.div_cont) ** self.dt - self.down) / (self.up - self.down)

    def __coupon(self, step):
        if step * self.dt % self.coupon_freq == 0:
            return self.coupon * self.principal
        else:
            return 0

    def __dividend(self, step):
        if step * self.dt % self.div_freq == 0:
            return self.div_disc
        else:
            return 0

    def __build_derivative(self):
        for i in range(self.size):
            self.tree[-1][i] = (self.tree[-1][i], self.__payoff_function(self.tree[-1][i]))
        for j in reversed(range(self.size-1)):
            c = self.__coupon(j)
            p = self.__prob(j)
            df = self.yc.df((j + 1) * self.dt, j * self.dt)
            print(j*self.dt, c)
            for i in range(j+1):
                rolling = df * (p * self.tree[j+1][i][1] + (1-p) * self.tree[j+1][i+1][1]) + c
                stock_at_con = (self.stock_out * self.tree[j][i] + self.cb_out * self.principal) / \
                               (self.stock_out + self.cb_out * self.cr)
                self.tree[j][i] = (self.tree[j][i], max(min(self.call, rolling), self.cr * stock_at_con))
        return None


if __name__ == "__main__":
    apple = ConvertibleTree(1000, 1, .0, .5, 4, "AAPL", 5)
    print(apple.tree)
    # pear = CRRTree(4, "AAPL", "call", 1, 200)
    # print(pear.tree[0])
