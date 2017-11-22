from dateutil.relativedelta import relativedelta as rel_delta
import datetime as dt
import math
import numpy as np
from scipy.stats import norm
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import quandl
import ecb
import re

style.use("ggplot")
quandl.ApiConfig.api_key = "MGThycVq2kfvk_VR1Jn_"


class YieldCurve(object):
    """
    cannot handle stress and spread at the same time
    every time (period) is measured in years
    """
    __stress_increase = np.array(   # Article 166
        [.7, .7, .7, .64, .59, .55, .52, .49, .47, .44, .42, .39, .37, .35, .34, .33, .31, .3, .29, .27, .26, .2])
    __stress_decrease = np.array(   # Article 167
        [.75, .75, .65, .56, .5, .46, .42, .39, .36, .33, .31, .3, .29, .28, .28, .27, .28, .28, .28, .29, .29, .2])
    __stress_tenors = np.array([x for x in range(21)] + [90])

    def __init__(self, date, stress, spread):
        if spread != .0 and stress is not None:
            print("""YieldCurve() cannot handle stress and spread simultaneously""")
            raise Exception
        self.date = date
        self.spread = spread
        self.stress = stress
        self.tenors = None  # [1 / 12, 1 / 4, 1 / 2, 1, 2, 3, 5., 7, 10, 20, 30]
        self.rates = None  # [0.96, 1.02, 1.1, 1.24, 1.35, 1.46, 1.73, 1.99, 2.16, 2.51, 2.77]
        self.data = None

    def __str__(self):
        return str(self.data)

    @staticmethod
    def __search_in(item, x):    # bisection search method between elements of an ordered list
        if x[0] <= item <= x[-1]:
            mn = 0
            mx = len(x) - 1
            j = 0
            if x[mn] == item:
                return mn
            if x[mx] == item:
                return mx
            while j <= len(x):
                if j == len(x):
                    print(""""search_in function failed in YieldCurve()""")
                    raise Exception
                md = int((mx + mn) / 2)
                if mn == md:
                    return None
                if x[md] == item:
                    return md
                elif x[md] > item:
                    mx = md
                    j += 1
                else:
                    mn = md
                    j += 1
        else:
            return None

    @staticmethod
    def __search_between(item, x):   # bisection search method between elements of an ordered list
        if x[0] < item < x[-1]:
            mn = 0
            mx = len(x) - 1
            j = 0
            while j <= len(x):
                if j == len(x):
                    print(""""search_between function failed in YieldCurve()""")
                    raise Exception
                md = int((mx + mn) / 2)
                if x[md] >= item:
                    mx = md
                    j += 1
                elif x[md + 1] < item:
                    mn = md + 1
                    j += 1
                else:
                    return md + 1
        else:
            return None

    def __approx(self, item, x, y):
        if x[0] < item < x[-1]:
            k = self.__search_between(item, x)
            return y[k - 1] + (item - x[k - 1]) * (y[k] - y[k - 1]) / (x[k] - x[k - 1])
        elif 0 < item < x[0]:
            return item * y[0] / x[0]
        elif item <= 0:
            return 0
        else:
            return x[-1]

    def __spot(self, maturity):
        k = self.__search_in(maturity, self.tenors)
        if isinstance(k, int):
            return self.rates[k] / 100
        else:
            return self.__approx(maturity, self.tenors, self.rates) / 100

    def __spot_stress(self, maturity, stress):
        if stress == "inc":
            m = 1 + self.__approx(maturity, self.__stress_tenors, self.__stress_increase)
            return max(self.__spot(maturity) * m, self.__spot(maturity) + 0.01)
        elif stress == "dec":
            m = 1 - self.__approx(maturity, self.__stress_tenors, self.__stress_decrease)
            if self.__spot(maturity) > 0:
                return self.__spot(maturity) * m
            else:
                return self.__spot(maturity)

    def __spot_spread(self, maturity, spread):
        return self.__spot(maturity) + spread

    def spot(self, maturity):
        if self.stress is not None:
            return self.__spot_stress(maturity, self.stress)
        elif self.spread != 0:
            return self.__spot_spread(maturity, self.spread)
        else:
            return self.__spot(maturity)

    def forward(self, closer, further):
        return ((1 + self.spot(further)) ** further / (1 + self.spot(closer)) ** closer) ** (1 / (further - closer)) - 1

    def df(self, maturity, base=0):
        return 1 / (1 + self.forward(base, maturity)) ** (maturity - base)

    def af(self, maturity, base=0, freq=1):
        def my_range(end, start, step):
            if start < end and step > 0:
                current = end
                while current > start:
                    yield current
                    current -= step

        s = 0
        for x in my_range(maturity, base, freq):
            s += self.df(x, base)
        return s

    def show(self):
        tenors = np.array([0, 1 / 12, 1 / 4, 1 / 2, 3 / 4] + [x for x in range(1, 31)])
        rates = np.array([self.__spot(x) for x in tenors])
        if self.stress is not None:
            curve_inc = np.array([self.__spot_stress(x, "inc") for x in tenors])
            curve_dec = np.array([self.__spot_stress(x, "dec") for x in tenors])
            plt.plot(tenors, rates, tenors, curve_inc, '--', tenors, curve_dec, '--')
        elif self.spread != 0:
            curve_spread = np.array([self.__spot_spread(x, self.spread) for x in tenors])
            plt.plot(tenors, rates, '--', tenors, curve_spread)
        else:
            plt.plot(tenors, rates)
        plt.show()
        return None


class UsTreasury(YieldCurve):
    """"
    spread is not implemented
    """
    def __init__(self, date=str(dt.date.today()), stress=None):
        super().__init__(date, stress, spread=.0)
        # get data from Quandl
        self.data = quandl.get("USTREASURY/YIELD", start_date=date, end_date=date)  # pandas DataFrame
        # get rates from data
        self.rates = self.data.values[0]
        # get tenors from data
        header = self.data.columns.values  # numpy n dim array
        self.tenors = np.empty(len(header), dtype=float)  # numpy n dim array
        for i, x in enumerate(header):
            # everything but the last three characters
            self.tenors[i] = float(x[:-3])
            # last two characters
            if x[-2:] == "MO":
                self.tenors[i] /= 12


class EuroArea(YieldCurve):
    def __init__(self, date=str(dt.date.today()), stress=None, spread=.0):
        super().__init__(date, stress, spread)
        # build tenors
        self.tenors = np.array([1 / 4, 1 / 2, 3 / 4] + [x for x in range(1, 31)])
        # get rates from ECB web
        if date == str(dt.date.today()):
            self.rates = ecb.Scraper().data()
            print("""
            Euro area yield curve from website of ECB \n
            """)
            for tenor, rate in zip(self.tenors, self.rates):
                print("\t", tenor, "\t\t", rate)
        else:
            self.rates = self.__local_parser(date)
        # build data
        self.data = dict(zip(self.tenors, self.rates))

    @staticmethod
    def __local_parser(date):
        tenor_checklist = ['3M', '6M', '9M'] + [str(i) + 'Y' for i in range(1, 31)]
        result = np.empty(len(tenor_checklist), dtype=float)
        with open("euro_area.csv", "r") as file:
            for line in file:
                date_search = re.search(str(date), line, re.I)
                if date_search:
                    name_search = re.search('[0-9]+Y|[369]M', line, re.I)
                    if name_search:
                        rate_search = re.search('[-]?[0-9]+[.][0-9]{4}', line)
                        result[tenor_checklist.index(name_search.group())] = rate_search.group()
        return result


class Stock(object):
    __stress_1 = .39
    __stress_2 = .49

    # symmetric_adj()

    def __init__(self, ticker, start=str(dt.date.today() - rel_delta(years=1)), end=str(dt.date.today()), stress=None):
        # get data from Yahoo finance
        table = None
        fail = True
        attempt = 1
        while fail:
            try:
                table = web.DataReader(ticker, 'yahoo', start, end)
                fail = False
            except Exception as e:
                print(str(attempt), "attempt failed \t", str(e))
        self.data = table['Adj Close']

        # get price from data
        if stress == "type1":
            price = float(self.data.values[-1]) * (1 - self.__stress_1 - self.symmetric_adj())
        elif stress == "type2":
            price = float(self.data.values[-1]) * (1 - self.__stress_2 - self.symmetric_adj())
        else:
            price = float(self.data.values[-1])
        self.price = price

    def __str__(self):
        return str(self.data)

    def vol_hist(self, days=252):
        n = min(days, len(self.data) - 1)
        if n == len(self.data) - 1:
            print("There is not enough data \n")
            print("Historical volatility is calculated from" + str(n) + "observation instead of" + str(days))
        returns = np.array([self.data.values[-x] / self.data.values[-x-1] - 1 for x in range(1, n+1)])
        return np.std(returns) * 252 ** 0.5

    @staticmethod
    def symmetric_adj():
        start = dt.date.today() - rel_delta(days=37)
        end = dt.date.today() - rel_delta(days=1)
        tickers = np.array(
            ['^AEX', '^FCHI', '^GDAXI', '^FTAS', 'FTSEMIB.MI', '^IBEX', '^SSMI', '^GSPC', '^OMX', '^N225'])
        weights = np.array([.14, .14, .14, .14, .08, .08, .02, .08, .08, .02])
        # get index data
        index = []
        for ticker in tickers:
            fail = True
            attempt = 1
            while fail:
                try:
                    index.append(web.DataReader(ticker, 'yahoo', start, end)['Adj Close'])
                    fail = False
                except Exception as e:
                    print(str(ticker), str(attempt), "attempt failed \t", str(e))
                    attempt += 1
        # calculate current index (CI) value
        ci = 0
        for x, w in zip(index, weights):
            ci += w * x[-1]
        # calculate average index (AI) value
        ai = 0
        for x, w in zip(index, weights):
            ai += w * np.mean(x)
        # calculate symmetric adjustment
        symmetric_adj = 0.5 * ((ci - ai) / ai - .08)
        return max(-.1, min(symmetric_adj, .1))


class Option(object):
    def __init__(self, ticker, strike=100, expiry='18-01-19', option='call'):
        self.ticker = ticker
        self.strike = strike
        self.expiry = expiry
        self.option_type = option

        self.table = web.Options(ticker, "yahoo").get_all_data()
        self.data = self.table['Last']
        self.price = float(self.data[strike, str(expiry), str(option)])
        self.maturity = (dt.datetime.strptime(self.expiry, "%y-%m-%d") - dt.datetime.today()) / dt.timedelta(days=365)
        self.expiry_dates = web.Options(ticker, "yahoo").expiry_dates

        self.stock = Stock(ticker)
        self.yield_curve = EuroArea(str(dt.date.today()))

    def __str__(self):
        return str(self.data)

    def __bs_diff(self, volatility):
        stock = self.stock.price
        strike = self.strike
        maturity = self.maturity
        r = math.ln(1 + self.yield_curve.spot(maturity))
        df = self.yield_curve.df(maturity)

        d1 = (math.ln(stock / strike) + (r + volatility ** 2 / 2) * maturity) / (volatility * maturity ** 0.5)
        d2 = d1 - volatility * maturity ** 0.5
        return stock * norm.cdf(d1) - strike * df * norm.cdf(d2) - self.price

    def __vega(self, volatility):
        stock = self.stock.price
        strike = self.strike
        maturity = self.maturity
        r = math.ln(1 + self.yield_curve.spot(maturity))
        d1 = (math.ln(stock / strike) + (r + volatility ** 2 / 2) * maturity) / (volatility * maturity ** 0.5)
        return stock * norm.pdf(d1) * maturity ** 0.5

    def vol_implied(self, epsilon=0.0001):
        guess = self.stock.vol_hist()
        delta = abs(0 - self.__bs_diff(guess))
        while delta > epsilon:
            guess -= self.__bs_diff(guess) / self.__vega(guess)
            delta = abs(0 - self.__bs_diff(guess))
        return guess


class Bond(object):
    __a = np.array([
        np.array([.0, .045, .07, .095, .12]),
        np.array([.0, .055, .084, .109, .134]),
        np.array([.0, .07, .105, .13, .155]),
        np.array([.0, .125, .2, .25, .3]),
        np.array([.0, .225, .35, .44, .465]),
        np.array([.0, .375, .585, .61, .635])
    ])
    __b = np.array([
        np.array([.009, .005, .005, .005, .005]),
        np.array([.011, .006, .005, .005, .005]),
        np.array([.014, .007, .005, .005, .005]),
        np.array([.025, .015, .01, .01, .005]),
        np.array([.045, .025, .018, .005, .005]),
        np.array([.075, .042, .005, .005, .005]),
    ])

    def __init__(self, principal, maturity, coupon, freq, yield_curve, stress=False, quality=0):
        self.principal = principal
        self.maturity = maturity
        self.coupon = coupon    # coupon rate
        self.freq = freq        # number of coupon payments in a year
        self.yield_curve = yield_curve

        if stress:
            self.price = (yield_curve.af(maturity, 0, freq) * coupon + yield_curve.df(maturity)) * principal \
                         * (1 - self.__stress(quality, self.modified_duration()))
        else:
            self.price = (yield_curve.af(maturity, 0, freq) * coupon + yield_curve.df(maturity)) * principal

    @staticmethod
    def __my_range(end, start, step):
        if start < end and step > 0:
            current = end
            while current > start:
                yield current
                current -= step

    def __ytm_diff(self, ytm):
        s = self.principal / (1 + ytm) ** self.maturity
        for t in self.__my_range(self.maturity, 0, self.freq):
            s += self.coupon * self.principal / (1 + ytm) ** t
        return s - self.price

    def __ytm_derivative(self, ytm):
        s = - self.principal * self.maturity * (1 + ytm) ** (- self.maturity - 1)
        for t in self.__my_range(self.maturity, 0, self.freq):
            s += - self.coupon * self.principal * t * (1 + ytm) ** (- t - 1)
        return s

    def ytm(self, epsilon=0.001):
        guess = (self.coupon * self.principal + (self.principal - self.price) / int(self.maturity / self.freq)) / \
                ((self.principal + self.price) / 2)
        delta = abs(0 - self.__ytm_diff(guess))
        while delta > epsilon:
            guess -= self.__ytm_diff(guess) / self.__ytm_derivative(guess)
            delta = abs(0 - self.__ytm_diff(guess))
        return guess

    def duration(self):
        s = self.maturity * self.principal * self.yield_curve.df(self.maturity)
        for t in self.__my_range(self.maturity, 0, self.freq):
            s += t * self.coupon * self.yield_curve.df(t)
        return s / self.price

    def modified_duration(self):
        return self.duration() / (1 + self.ytm() / self.freq)

    def __stress(self, quality, duration):
        if duration <= 5:
            x = duration
            dur = 0
        elif 5 < duration <= 10:
            x = duration - 5
            dur = 1
        elif 10 < duration <= 15:
            x = duration - 10
            dur = 2
        elif 15 < duration <= 20:
            x = duration - 15
            dur = 3
        else:
            x = duration - 20
            dur = 4
        return self.__a[quality][dur] + self.__b[quality][dur] * x


if __name__ == '__main__':
    my_yield_curve = EuroArea()
