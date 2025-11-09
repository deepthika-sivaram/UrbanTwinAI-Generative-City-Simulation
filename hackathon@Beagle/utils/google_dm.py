# utils/google_dm.py
import os, requests

API_KEY = "AIzaSyBZu_fK0_ydly6-SR5NF2wm-sjveQ6eSbQ"
URL = "https://maps.googleapis.com/maps/api/distancematrix/json"

def dist_matrix(origin_lonlat, dest_lonlat, departure_time="now", traffic_model="best_guess"):
    params = {
        "origins": f"{origin_lonlat[1]},{origin_lonlat[0]}",
        "destinations": f"{dest_lonlat[1]},{dest_lonlat[0]}",
        "departure_time": departure_time,
        "traffic_model": traffic_model,
        "key": API_KEY,
    }
    r = requests.get(URL, params=params, timeout=10); r.raise_for_status()
    el = r.json()["rows"][0]["elements"][0]
    if el.get("status") != "OK": return None
    t  = el["duration"]["value"]
    tt = el.get("duration_in_traffic", {}).get("value", t)
    return t, tt


'''
AIzaSyBZu_fK0_ydly6-SR5NF2wm-sjveQ6eSbQ

'''