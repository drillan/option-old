import numpy as np
import pandas as pd
import ivolat3


def get_itm(df, s, t):
    itm_df = df.copy()
    itm_df["ITM"] = itm_df["Call/Put"].map(lambda x: {"C": "P", "P": "C"}[x])
    itm_price = itm_df.apply(
        lambda x: ivolat3.prem(s, x.name, r, 0, t, x["iv"], x["ITM"]), axis=1
    )
    return pd.Series(itm_price.values, index=zip(itm_df.index, itm_df["ITM"]))


s_df = pd.read_pickle("./data/s.pickle")
op_pn1 = pd.read_pickle("./data/op1.pickle")
op_pn2 = pd.read_pickle("./data/op2.pickle")
r = 0.001


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
