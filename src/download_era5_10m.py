"""Download hourly ERA5 10 m wind speed for the 20 candidate sites (used with the
100 m series to derive hourly shear exponents and extrapolate to the 80 m hub height)."""
import json
import os
import time
import requests

SITES = {
    "T01": (15.993, -94.428), "T02": (15.993, -94.683), "T03": (15.993, -94.939),
    "T04": (15.993, -95.109), "T05": (16.063, -95.369), "T06": (16.274, -94.444),
    "T07": (16.274, -94.530), "T08": (16.344, -94.791), "T09": (16.274, -95.128),
    "T10": (16.274, -95.385), "T11": (16.485, -94.371), "T12": (16.485, -94.629),
    "T13": (16.485, -94.886), "T14": (16.485, -95.143), "T15": (16.485, -95.400),
    "T16": (16.766, -94.388), "T17": (16.766, -94.560), "T18": (16.766, -94.904),
    "T19": (16.766, -95.163), "T20": (16.696, -95.415),
}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_10m")
os.makedirs(OUT_DIR, exist_ok=True)
URL = "https://archive-api.open-meteo.com/v1/archive"

for sid, (lat, lon) in SITES.items():
    path = os.path.join(OUT_DIR, f"{sid}_{lat}_{lon}.json")
    if os.path.exists(path):
        continue
    for attempt in range(6):
        r = requests.get(URL, params={"latitude": lat, "longitude": lon, "start_date": "2000-01-01",
                                       "end_date": "2024-12-31", "hourly": "wind_speed_10m",
                                       "wind_speed_unit": "ms", "timezone": "UTC"}, timeout=120)
        if r.status_code == 200:
            json.dump(r.json(), open(path, "w"))
            print("ok", sid)
            break
        print("retry", sid, r.status_code)
        time.sleep(15 * (attempt + 1))
    time.sleep(2)
print("done")
