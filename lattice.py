import datetime as dt
import numpy as np
import math
from bsmarket import Option


class BinomialTree(object):
    def __init__(self, steps):
        self.size = steps + 1
        self.data = [np.empty(x+1, dtype=tuple) for x in range(self.size)]

    def __str__(self):
        return str(self.data)


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
        self.up = math.exp(self.volatility * self.dt ** 0.5)
        self.down = 1 / self.up

        # build stock tree
        self.__build_stock()

        # build option tree
        print(self.option.price)
        print(self.option.maturity)
        self.__build_derivative()

    def __build_stock(self):
        self.data[0][0] = self.stock.price
        for i in range(1, self.size):
            for j in range(i, self.size):
                self.data[j][i - 1] = self.data[j - 1][i - 1] * self.up
            self.data[i][i] = self.data[i - 1][i - 1] * self.down
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
        f = self.yc.forward(step, step + self.dt)
        return ((1 + f) ** self.dt - self.down) / (self.up - self.down)

    def __build_derivative(self):
        for i in range(self.size):
            self.data[-1][i] = (self.data[-1][i], self.__payoff_function(self.data[-1][i]))
        for j in reversed(range(self.size-1)):
            p = self.__prob(j*self.dt)
            df = self.yc.df((j+1)*self.dt, j*self.dt)
            for i in range(j+1):
                self.data[j][i] = (self.data[j][i], df * (p * self.data[j+1][i][1] + (1-p) * self.data[j+1][i+1][1]))


class ConvertibleTree(BinomialTree):

    def __init__(self, principal, maturity, coupon, freq, steps, ticker, conversion_ratio):
        # first_coupon = maturity % freq
        # max_dt = math.gcd(first_coupon, freq)
        # if max_dt < maturity / steps:
        #     print("Number of steps is not enough")
        #     raise Exception
        # pass input to parent
        super().__init__(steps)

        # save inputs
        self.principal = principal
        self.maturity = maturity        # years
        self.coupon = coupon
        self.freq = freq
        self.steps = steps
        self.ticker = ticker
        self.cr = conversion_ratio

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
        self.up = math.exp(self.volatility * self.dt ** 0.5)
        self.down = 1 / self.up

        # build stock tree
        self.__build_stock()

        # build option tree
        print(self.option.price)
        print(self.option.maturity)
        self.__build_derivative()

    def __build_stock(self):
        self.data[0][0] = self.stock.price
        for i in range(1, self.size):
            for j in range(i, self.size):
                self.data[j][i - 1] = self.data[j - 1][i - 1] * self.up
            self.data[i][i] = self.data[i - 1][i - 1] * self.down
        return None

    def __payoff_function(self, stock):
        return max((1 + self.coupon) * self.principal, self.cr * stock)

    def __prob(self, step):
        f = self.yc.forward(step, step + self.dt)
        print("forward: " + str(f))
        return ((1 + f) ** self.dt - self.down) / (self.up - self.down)

    def __coupon(self, step):
        if step % self.freq == 0:
            return self.coupon * self.principal
        else:
            return 0

    def __build_derivative(self):
        for i in range(self.size):
            self.data[-1][i] = (self.data[-1][i], self.__payoff_function(self.data[-1][i]))
        for j in reversed(range(self.size-1)):
            c = self.__coupon(j*self.dt)
            p = self.__prob(j*self.dt)
            df = self.yc.df((j+1)*self.dt, j*self.dt)
            print(j*self.dt, c)
            for i in range(j+1):
                if self.cr * self.data[j][i] > df * (p * self.data[j+1][i][1] + (1-p) * self.data[j+1][i+1][1]) + c:
                    print("Early conversion")
                self.data[j][i] = (self.data[j][i], max(df * (p * self.data[j+1][i][1] + (1-p) * self.data[j+1][i+1][1])
                                                        + c, self.cr * self.data[j][i]))


if __name__ == "__main__":
    apple = ConvertibleTree(1000, 1, .0, .5, 4, "AAPL", 5)
    print(apple.data[0])
    # pear = CRRTree(4, "AAPL", "call", 1, 200)
    # print(pear.data[0])
