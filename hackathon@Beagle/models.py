def apply_scenario(df, add_build_pct=0, add_green_pct=0):
    d = df.copy()
    d["building_cov"] = (d["building_cov"] * (1 + add_build_pct/100)).clip(0,1)
    d["green_cov"] = (d["green_cov"] * (1 + add_green_pct/100)).clip(0,1)
    d["impervious"] = (d["building_cov"] + d["road_den"]*0.02).clip(0,1)
    return d

def uhi_delta(df, a1=6.0, a2=4.0, a3=1.0, water_adj=0.0):
    uhi = a1*df["impervious"] - a2*df["green_cov"] - a3*water_adj
    return (uhi - uhi.mean())

def traffic_delay_pct(df, b1=0.6, b2=0.3, b3=0.2):
    raw = b1*df["road_den"] + b2*df["int_den"] + b3*df["building_cov"]
    return 100*(raw - raw.mean())

def pm25_delta(df, c0=7.0, c1=0.03, c2=3.0):
    delay = traffic_delay_pct(df)/100
    pm = c0 + c1*delay + c2*(1 - df["green_cov"])
    return (pm - pm.mean())

def uhi_raw(df, a1=6.0, a2=4.0, a3=1.0, water_adj=0.0):
    return a1*df["impervious"] - a2*df["green_cov"] - a3*water_adj

def traffic_delay_raw(df, b1=0.6, b2=0.3, b3=0.2):
    return b1*df["road_den"] + b2*df["int_den"] + b3*df["building_cov"]

def pm25_raw(df, c0=7.0, c1=0.03, c2=3.0):
    delay = traffic_delay_raw(df)
    return c0 + c1*delay + c2*(1 - df["green_cov"])
