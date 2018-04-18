import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def payoff_fut(qty, s, p, k=None):
    return (s - p) * qty


def payoff_call(qty, s, p, k):
    return (max(s - k, 0) - p) * qty


def payoff_put(qty, s, p, k):
    return (max(k - s, 0) - p) * qty


def payoff(right, qty, s, p, k=None):
    func = {'s': payoff_fut, 'c': payoff_call, 'p': payoff_put}
    return func[right](qty, s, p, k)


class Portfolio:
    mergin_pct = 0.05
    linspace_num = 100

    def __init__(self, position):
        self.df = pd.DataFrame([], columns=['qty', 'right', 'k', 'p'])
        self.position_name = []
        for i, p in enumerate(position):
            if not 'k' in p:
                p['k'] = None
            
            if p['k']:
                pname = '{}{} x {}@{}'.format(p['k'], p['right'], p['qty'], p['p'])
            else:
                pname = '{} x {}@{}'.format(p['right'], p['qty'], p['p'])
                
            self.df.loc[i] = pd.Series(p)
            self.position_name.append(pname)

        if self.df['k'].any():
            max_ = self.df['k'].max()
            min_ = self.df['k'].min()
        else:
            max_ = self.df['p'].max()
            min_ = self.df['p'].min()

        mergin = min_ * (1 - self.mergin_pct), max_ * (1 + self.mergin_pct)
        self.s_range = np.linspace(*mergin, self.linspace_num)
        returns = [self.make_payoff_range(self.df.loc[i])
                   for i in self.df.index]
        returns = pd.DataFrame(returns).T
        returns.index = self.s_range
        self.returns = returns
        self.returns_sum = returns.sum(axis=1)

    def make_payoff_range(self, ser):
        return pd.Series([payoff(**ser.to_dict(), s=s) for s in self.s_range], index=self.s_range)
    
    def plot_payoff(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.stackplot(self.s_range, self.returns_sum)
        for i in self.returns.columns:
            ax.plot(self.s_range, self.returns[i], label=self.position_name[i])
        ax.legend()
        plt.show()


def plot_payoff(positions):
    def str_to_dict(position):
        k_right, qty_price = position.split('x')
        if k_right == 's':
            k = None
            right = 's'
        else:
            k = float(k_right[:-1])
            right = k_right[-1]
        qty, price = qty_price.split('@')
        qty, price = float(qty), float(price)
        return {'qty': qty, 'right': right, 'k': k, 'p': price}

    positions = [str_to_dict(x) for x in positions]
    p = Portfolio(positions)
    p.plot_payoff()

