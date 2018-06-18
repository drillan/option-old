from io import BytesIO
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML
from jinja2 import Template
import ivolat3


s_df = pd.read_pickle("./data/s.pickle")
op_pn1 = pd.read_pickle("./data/op1.pickle")
op_pn2 = pd.read_pickle("./data/op2.pickle")
r = 0.001
q = 0
const_1per365 = 1 / 365


def get_itm(df, s, t):
    itm_df = df.copy()
    itm_df["ITM"] = itm_df["Call/Put"].map(lambda x: {"C": "P", "P": "C"}[x])
    itm_price = itm_df.apply(
        lambda x: ivolat3.prem(s, x.name, r, 0, t, x["iv"], x["ITM"]), axis=1
    )
    return pd.Series(itm_price.values, index=zip(itm_df.index, itm_df["ITM"]))


def get_fop_data(t0):
    t0 = pd.to_datetime(t0)
    t1_1 = pd.to_datetime("2018-03-09 09:00")
    t1_2 = pd.to_datetime("2018-04-13 09:00")
    s0_1 = s_df.loc[t0, "1803"]
    s0_2 = s_df.loc[t0, "1804"]
    delta_t_1 = t1_1 - t0
    delta_t_2 = t1_2 - t0
    t_1 = (delta_t_1.days / 365) + (delta_t_1.seconds / 31536000)
    t_2 = (delta_t_2.days / 365) + (delta_t_2.seconds / 31536000)
    op_df1 = op_pn1.loc[t0].dropna().reset_index().copy()
    op_df2 = op_pn2.loc[t0].dropna().reset_index().copy()
    op_df1.set_index("k", inplace=True)
    op_df2.set_index("k", inplace=True)
    op_df1[["price", "iv"]] = op_df1[["price", "iv"]].astype(float)
    op_df2[["price", "iv"]] = op_df2[["price", "iv"]].astype(float)
    op_df1 = op_df1.reindex(
        np.arange(op_df1.index.min(), op_df1.index.max() + 125, 125)
    ).copy()
    op_df2 = op_df2.reindex(
        np.arange(op_df2.index.min(), op_df2.index.max() + 125, 125)
    ).copy()
    op_df1.loc[op_df1.index <= s0_1, "Call/Put"] = "P"
    op_df1.loc[op_df1.index > s0_1, "Call/Put"] = "C"
    op_df2.loc[op_df2.index <= s0_2, "Call/Put"] = "P"
    op_df2.loc[op_df2.index > s0_2, "Call/Put"] = "C"
    op_df1["iv"] = op_df1["iv"].interpolate(method="cubic")
    op_df2["iv"] = op_df2["iv"].interpolate(method="cubic")
    op_df1.loc[op_df1["price"].isnull(), "price"] = op_df1.loc[
        op_df1["price"].isnull()
    ].apply(
        lambda x: ivolat3.prem(s0_1, x.name, r, 0, t_1, x["iv"], x["Call/Put"]), axis=1
    )
    op_df2.loc[op_df2["price"].isnull(), "price"] = op_df2.loc[
        op_df2["price"].isnull()
    ].apply(
        lambda x: ivolat3.prem(s0_2, x.name, r, 0, t_2, x["iv"], x["Call/Put"]), axis=1
    )
    otm_price1 = pd.Series(
        op_df1["price"].values, index=zip(op_df1.index, op_df1["Call/Put"])
    )
    otm_price2 = pd.Series(
        op_df2["price"].values, index=zip(op_df2.index, op_df2["Call/Put"])
    )
    itm_price1 = get_itm(op_df1, s0_1, t_1)
    itm_price2 = get_itm(op_df2, s0_2, t_2)
    full_price1 = pd.concat([otm_price1, itm_price1])
    full_price2 = pd.concat([otm_price2, itm_price2])
    return (s0_1, t_1, op_df1, full_price1), (s0_2, t_2, op_df2, full_price2)


class Portfolio:
    maturity_dict = {0: "1803", 1: "1804"}

    def __init__(self):
        self.id = 0
        self.position = pd.DataFrame(
            [],
            columns=[
                "code",
                "t0",
                "qty",
                "maturity",
                "k",
                "right",
                "a_price",
                "c_price",
                "pl",
                "iv",
                "delta",
                "gamma",
                "vega",
                "theta",
            ],
        )

    def add(self, t0, qty, maturity, right, k=np.nan):
        if right == "F":
            code = "{}{}".format(self.maturity_dict[maturity], right)
        else:
            code = "{}{}{}".format(self.maturity_dict[maturity], right, k)

        self.position.loc[
            self.id, ["code", "t0", "qty", "maturity", "k", "right", "pl"]
        ] = (code, t0, qty, maturity, k, right, 0)

        price, iv, delta, gamma, vega, theta = self.get_info(
            t0, qty, maturity, right, k
        )

        self.position.loc[
            self.id, ["a_price", "iv", "delta", "gamma", "vega", "theta"]
        ] = (price, iv, delta, gamma, vega, theta)

        self.id += 1

    def update(self, t0):
        current = self.position[["qty", "maturity", "right", "k"]].apply(
            lambda x: pd.Series(self.get_info(t0, *x)), axis=1
        )
        self.position[["c_price", "iv", "delta", "gamma", "vega", "theta"]] = current
        pl = (self.position["c_price"] - self.position["a_price"]) * self.position[
            "qty"
        ]
        self.position["pl"] = pl

    def get_info(self, t0, qty, maturity, right, k):
        data1, data2 = get_fop_data(t0)
        data = list(zip(data1, data2))
        s0, t, op_df, full_price = data

        if right == "F":
            price = s_df.loc[t0, self.maturity_dict[maturity]]
            iv = np.nan
            delta = qty
            gamma, vega, theta = 0.0, 0.0, 0.0
        else:
            price = full_price[maturity][k, right]
            iv = op_df[maturity].loc[k, "iv"]
            s_, t_ = s0[maturity], t[maturity]
            delta = ivolat3.delta(s_, k, r, q, t_, iv, right) * qty
            gamma = ivolat3.gamma(s_, k, r, q, t_, iv) * qty * 1000
            vega = ivolat3.vega(s_, k, r, q, t_, iv) * qty
            theta = ivolat3.theta(s_, k, r, q, t_, iv, right) * qty * const_1per365
        return price, iv, delta, gamma, vega, theta


def plot_iv(t0, axes, prev1=None, prev2=None):
    (s0_1, t_1, op_df1, full_price1), (s0_2, t_2, op_df2, full_price2) = get_fop_data(
        t0
    )
    axes.plot(op_df1.index, op_df1["iv"], color="#1f77b4")
    axes.plot(op_df2.index, op_df2["iv"], color="#ff7f0e")
    axes.set_xlim(12500, 26000)
    axes.set_ylim(0.15, 1)
    yticks = axes.axes.get_yticks()

    s0_1_prev = np.nan
    if prev1 and prev2:
        (s0_1_prev, t_1_prev, op_df1_prev), (s0_2_prev, t_2_prev, op_df2_prev) = (
            prev1,
            prev2,
        )
        axes.plot(op_df1_prev.index, op_df1_prev["iv"], linestyle="--", color="#1f77b4")
        axes.plot(op_df2_prev.index, op_df2_prev["iv"], linestyle="--", color="#ff7f0e")
        yticks = axes.axes.get_yticks()
        axes.vlines(s0_1_prev, yticks[0], yticks[-1], linestyle="--", linewidth=1)
        axes.set_title(
            "{:%m-%d} f1:{} -> {}({:+.2f})".format(
                t0, s0_1_prev, s0_1, s0_1 - s0_1_prev
            )
        )
    else:
        axes.set_title("{:%m-%d} f1:{}".format(t0, s0_1))

    axes.vlines(s0_1, yticks[0], yticks[-1], linestyle="-", linewidth=1)
    return (s0_1, t_1, op_df1), (s0_2, t_2, op_df2)


def draw_dashboard(portofolio, t0, prev1=None, prev2=None):
    html_tpl = """
    <table>
    <tr colspan="2">
        <td colspan="2">
        {{ position }}
        </td>
    <tr>
        <td><img src="data:image/png;base64,{{ png_data }}" /></td>
        <td>{{ summary }}</td>
    </tr>
    </table>
"""
    bytes_data = BytesIO()
    plt.ioff()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    prev1, prev2 = plot_iv(t0, ax1, prev1, prev2)
    fig.savefig(bytes_data, format="png")
    plt.close(fig)
    png_data = base64.b64encode(bytes_data.getvalue()).decode()
    tpl = Template(html_tpl)
    position = portofolio.position[
        [
            "qty",
            "maturity",
            "right",
            "a_price",
            "c_price",
            "pl",
            "iv",
            "delta",
            "gamma",
            "vega",
            "theta",
        ]
    ]
    summary = pd.DataFrame(position.sum()[["pl", "delta", "gamma", "vega", "theta"]])
    html_text = tpl.render({"position": position.to_html(), "png_data": png_data, "summary": summary.to_html()})
    return HTML(html_text), prev1, prev2

