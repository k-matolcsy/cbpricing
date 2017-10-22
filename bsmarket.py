from dateutil.relativedelta import relativedelta as rel_delta
import datetime as dt
import math
import numpy as np
from scipy.stats import norm
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import quandl
import re

style.use("ggplot")
quandl.ApiConfig.api_key = "MGThycVq2kfvk_VR1Jn_"


class YieldCurve(object):
    """
    US zero-coupon yield curve
    Domain range is from 0 to 30YR
    """
    __stress_increase = np.array(
        [.7, .7, .7, .64, .59, .55, .52, .49, .47, .44, .42, .39, .37, .35, .34, .33, .31, .3, .29, .27, .26, .2])
    __stress_decrease = np.array(
        [.75, .75, .65, .56, .5, .46, .42, .39, .36, .33, .31, .3, .29, .28, .28, .27, .28, .28, .28, .29, .29, .2])
    __stress_axis = np.array([x for x in range(21)] + [90])

    def __init__(self, date, stress=None):
        """
        :param date: example "2017-09-05"
        """
        assert type(date) is str
        self.date = date
        self.stress = stress
        self.data = None
        self.data_raw = None  # [0.96, 1.02, 1.1, 1.24, 1.35, 1.46, 1.73, 1.99, 2.16, 2.51, 2.77]
        self.axis = None  # [1 / 12, 1 / 4, 1 / 2, 1, 2, 3, 5., 7, 10, 20, 30]

    def __str__(self):
        return str(self.data)

    @staticmethod
    def __search_in(time, axis):
        mn = 0
        mx = len(axis) - 1
        j = 0
        if axis[mn] == time:
            return mn
        if axis[mx] == time:
            return mx
        while j < len(axis):
            md = int((mx + mn) / 2)
            if mn == md:
                return None
            if time == axis[md]:
                return md
            elif time < axis[md]:
                mx = md
                j += 1
            else:
                mn = md
                j += 1

    @staticmethod
    def __search_between(time, axis):  # bisection search method between elements of an ordered list
        if axis[0] < time < axis[-1]:
            mn = 0
            mx = len(axis) - 1
            j = 0
            while j < len(axis):
                md = int((mx + mn) / 2)
                if axis[md] >= time:
                    mx = md
                    j += 1
                elif axis[md + 1] < time:
                    mn = md + 1
                    j += 1
                else:
                    return md + 1
        else:
            return None

    def __extrapolation(self, time, axis, data):
        if axis[0] < time < axis[-1]:
            k = self.__search_between(time, axis)
            return data[k - 1] + (time - axis[k - 1]) * (data[k] - data[k - 1]) / (axis[k] - axis[k - 1])
        elif 0 < time < axis[0]:
            return time * data[0] / axis[0]
        elif time <= 0:
            return 0
        else:
            return axis[-1]

    def __spot(self, time):
        axis = self.axis
        data = self.data_raw
        k = self.__search_in(time, axis)
        if isinstance(k, int):
            return data[k] / 100
        else:
            return self.__extrapolation(time, axis, data) / 100

    def __spot_stress(self, time, stress):
        if stress is None:
            return self.__spot(time)
        elif stress == "inc":
            m = 1 + self.__extrapolation(time, YieldCurve.__stress_axis, YieldCurve.__stress_increase)
            return max(self.__spot(time) * m, self.__spot(time) + 0.01)
        elif stress == "dec":
            m = 1 + self.__extrapolation(time, YieldCurve.__stress_axis, YieldCurve.__stress_decrease)
            if self.__spot(time) > 0:
                return self.__spot(time) / m
            else:
                return self.__spot(time)

    def spot(self, time):
        stress = self.stress
        if stress == "inc":
            m = 1 + self.__extrapolation(time, YieldCurve.__stress_axis, YieldCurve.__stress_increase)
            return max(self.__spot(time) * m, self.__spot(time) + 0.01)
        elif stress == "dec":
            m = 1 + self.__extrapolation(time, YieldCurve.__stress_axis, YieldCurve.__stress_decrease)
            if self.__spot(time) > 0:
                return self.__spot(time) / m
            else:
                return self.__spot(time)
        else:
            return self.__spot(time)

    def forward(self, start, end):
        return ((1 + self.spot(end)) ** end / (1 + self.spot(start)) ** start) ** (1 / (end - start)) - 1

    def df(self, end, start=0):
        return 1 / (1 + self.forward(start, end)) ** (end - start)

    def af(self, end, start=0, step=1):
        def my_range(end_range, start_range, step_range):
            if start_range < end_range and step_range > 0:
                current = end_range
                while current > start_range:
                    yield current
                    current -= step_range
        s = 0
        for x in my_range(end, start, step):
            s += self.df(x, start)
        return s

    def show(self):
        axis = np.array([0, 1/12, 1/4, 1/2, 3/4] + [x for x in range(1, 31)])
        curve = np.array([self.__spot(x) for x in axis])
        curve_inc = np.array([self.__spot_stress(x, "inc") for x in axis])
        curve_dec = np.array([self.__spot_stress(x, "dec") for x in axis])
        plt.plot(axis, curve, axis, curve_inc, '--', axis, curve_dec, '--')
        plt.show()
        return None


class UsTreasury(YieldCurve):
    def __init__(self, date, stress=None):
        super().__init__(date, stress)
        self.data = quandl.get("USTREASURY/YIELD", start_date=date, end_date=date)  # pandas DataFrame
        self.data_raw = self.data.values[0]
        # time-axis
        head = self.data.columns.values  # numpy n dim array
        size = len(head)
        self.axis = np.empty(size, dtype=float)  # numpy n dim array
        for i, x in enumerate(head):
            # everything but the last three characters
            self.axis[i] = float(x[:-3])
            # last two characters
            if x[-2:] == "MO":
                self.axis[i] /= 12


class EuroArea(YieldCurve):
    def __init__(self, date, stress=None):
        super().__init__(date, stress)
        self.data_raw = self.__parser(date)
        self.axis = np.array([1/4, 1/2, 3/4] + [x for x in range(1, 31)])
        self.data = dict(zip(self.axis, self.data_raw))

    @staticmethod
    def __parser(date):
        checklist = ['3M', '6M', '9M'] + [str(i) + 'Y' for i in range(1, 31)]
        data = np.empty(len(checklist), dtype=float)
        with open("euro_area.csv", "r") as file:
            for line in file:
                date_search = re.search(str(date), line, re.I)
                if date_search:
                    name_search = re.search('[0-9]+Y|[369]M', line, re.I)
                    if name_search:
                        rate_search = re.search('[-]?[0-9]+[.][0-9]{4}', line)
                        data[checklist.index(name_search.group())] = rate_search.group()
        return data


class SymmetricAdjIndex(object):
    __start = dt.date.today()-rel_delta(days=37)
    __end = dt.date.today()-rel_delta(days=1)
    __tickers = np.array(['^AEX', '^FCHI', '^GDAXI', '^FTAS', 'FTSEMIB.MI', '^IBEX', '^SSMI', '^GSPC', '^OMX', '^N225'])
    __weights = np.array([.14, .14, .14, .14, .08, .08, .02, .08, .08, .02])

    index = []
    for ticker in __tickers:
        fail = True
        attempt = 1
        while fail:
            try:
                index.append(web.DataReader(ticker, 'yahoo', __start, __end)['Adj Close'])
                fail = False
            except Exception as e:
                print(str(ticker), str(attempt), "attempt failed    ", str(e))
                attempt += 1

    ci = 0
    for x, w in zip(index, __weights):
        ci += w * x[-1]

    ai = 0
    for x, w in zip(index, __weights):
        ai += w * np.mean(x)

    symmetric_adj_raw = 0.5 * ((ci - ai) / ai - .08)
    symmetric_adj = max(-.01, min(symmetric_adj_raw, .1))


class Stock(object):
    __stress_1 = .39
    __stress_2 = .49
    # symmetric_adj

    def __init__(self, ticker, start=str(dt.date.today()-rel_delta(years=1)), end=str(dt.date.today()), stress=None):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.stress = stress

        fail = True
        attempt = 1
        while fail:
            try:
                self.table = web.DataReader(ticker, 'yahoo', start, end)
                fail = False
            except Exception as e:
                print(str(attempt), "failed     ", str(e))
        self.data = self.table['Adj Close']

        if stress == "type1":
            self.price = float(self.data.values[-1]) * (1 - self.__stress_1 - SymmetricAdjIndex().symmetric_adj)
        elif stress == "type2":
            self.price = float(self.data.values[-1]) * (1 - self.__stress_2 - SymmetricAdjIndex().symmetric_adj)
        else:
            self.price = float(self.data.values[-1])

    def __str__(self):
        return str(self.data)

    def vol_hist(self, days=252):
        n = min(days, len(self.data) - 1)
        data = np.array([self.data.values[-x] / self.data.values[-x-1] - 1 for x in range(1, n+1)])
        std_deviation = np.std(data)
        return std_deviation * 252 ** 0.5


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

        d1 = (math.ln(stock/strike) + (r + volatility ** 2 / 2) * maturity) / (volatility * maturity ** 0.5)
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
    def __init__(self, principal, maturity, coupon, freq, spread, yield_curve):
        self.principal = principal
        self.maturity = maturity
        self.coupon = coupon
        self.freq = freq
        self.spread = spread
        self.yield_curve = yield_curve

        self.price = (yield_curve.af(maturity, 0, freq) * coupon + yield_curve.df(maturity)) * principal
        # self.price_c = yield_curve.af(maturity, 0, freq) * coupon * principal
        # self.price_p = yield_curve.df(maturity) * principal

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


if __name__ == '__main__':
    pass
