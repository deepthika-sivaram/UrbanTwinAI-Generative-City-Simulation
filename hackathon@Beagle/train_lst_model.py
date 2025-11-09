import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
from sklearn.linear_model import LinearRegression
import joblib
from geo_utils import fetch_osm, grid_bbox, features 

DATA_DIR = "data"  
MODEL_PATH = "lst_model.pkl"

def extract_features_from_bbox(minx, miny, maxx, maxy):
    print(f"Fetching OSM for bbox: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")
    try:
        bld, roads, green, poly_m = fetch_osm((minx, miny, maxx, maxy))
        grid = grid_bbox(poly_m, cell=50)
        LIMIT_TILES = 50
        grid = grid[:LIMIT_TILES]

        g = features(grid, bld, roads, green)

        return {
            "building_count": int((g["building_cov"] > 0.1).sum()),
            "greenery_percent": float((g["green_cov"].mean()) * 100)
        }
    except Exception as e:
        print(f"Failed to extract features for bbox: {e}")
        return None

def average_lst_from_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        data[data < -50] = np.nan
        data[data > 70] = np.nan

        if np.isnan(data).all():
            print(f"⚠️ {os.path.basename(path)}: all values invalid or masked.")
            return None

        mean_val = float(np.nanmean(data))
        print(f"{os.path.basename(path)} → Mean LST: {mean_val:.2f}°C")
        return mean_val

def train_model_tile_level(tile_size=0.01): 
    rows = []
    for tif_path in glob.glob(os.path.join(DATA_DIR, "lst_*.tif")):
        city = os.path.basename(tif_path).replace("lst_", "").replace(".tif", "")
        print(f"\nProcessing {city}")

        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            transform = src.transform

            minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
            x_coords = np.arange(minx, maxx, tile_size)
            y_coords = np.arange(miny, maxy, tile_size)
            tile_count = 0
            for x in x_coords:
                for y in y_coords:
                    if tile_count >= 50:
                        break
                    tile_box = box(x, y, x + tile_size, y + tile_size)
                    geom = [mapping(tile_box)]

                    try:
                        lst_data, _ = mask(src, geom, crop=True)
                        data = lst_data[0].astype("float32")

                        if src.nodata is not None:
                            data[data == src.nodata] = np.nan
                        data[data < -50] = np.nan
                        data[data > 70] = np.nan

                        if np.isnan(data).all():
                            continue  

                        avg_lst = float(np.nanmean(data))

                        tile_bounds = tile_box.bounds
                        features = extract_features_from_bbox(*tile_bounds)
                        if features is None or 'building_count' not in features:
                            continue

                        rows.append({
                            "city": city,
                            "tile_x": x,
                            "tile_y": y,
                            "building_count": features["building_count"],
                            "greenery_percent": features["greenery_percent"],
                            "lst_celsius": avg_lst
                        })
                        tile_count += 1
                    except Exception as e:
                        print(f"⚠️ Skipping tile ({x:.3f}, {y:.3f}): {e}")
                        continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("No tile-level data extracted.")
        return

    print(f"\nTile-level dataset: {len(df)} samples")
    print(df.head())

    X = df[["building_count", "greenery_percent"]]
    y = df["lst_celsius"]

    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    print(f"\nTile-level model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model_tile_level()
