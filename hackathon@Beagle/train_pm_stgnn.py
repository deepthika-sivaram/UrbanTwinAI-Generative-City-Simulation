import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from stgnn_pm import FEATS 

rng = np.random.default_rng(42)
DEVICE = "cpu"  

def synth_batch(batch_cells=196, batches=800):

    X_list, y_list = [], []
    for _ in range(batches):
        building_cov = rng.beta(2, 5, batch_cells)
        green_cov    = rng.beta(3, 3, batch_cells)
        road_den     = rng.gamma(1.5, 0.002, batch_cells)
        impervious   = np.clip(building_cov + 0.02*road_den, 0, 1)
        traffic_lvl  = np.clip(0.5*road_den/(road_den.max() + 1e-9) + 0.1*rng.random(batch_cells), 0, 1)
        uhi_raw      = 6*impervious - 4*green_cov

        X = np.column_stack([building_cov, green_cov, road_den, impervious, traffic_lvl, uhi_raw])

        noise = rng.normal(0.0, 0.25, batch_cells)
        pm = 7.0 + 3.0*traffic_lvl + 2.5*(1.0 - green_cov) + 0.4*uhi_raw + noise

        X_list.append(X.astype(np.float32))
        y_list.append(pm.astype(np.float32))

    X_all = np.vstack(X_list)  
    y_all = np.hstack(y_list)  
    return X_all, y_all

def build_mlp(in_dim: int):
    return nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )

def main():
    os.makedirs("artifacts", exist_ok=True)

    X, y = synth_batch(batch_cells=14*14, batches=800)  # ~156k samples

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = build_mlp(len(FEATS)).to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    X_t = torch.from_numpy(Xs).float().to(DEVICE)
    y_t = torch.from_numpy(y).float().to(DEVICE)

    n = X_t.size(0)
    n_train = int(0.9 * n)
    perm = torch.randperm(n)
    tr_idx, va_idx = perm[:n_train], perm[n_train:]
    X_tr, y_tr = X_t[tr_idx], y_t[tr_idx]
    X_va, y_va = X_t[va_idx], y_t[va_idx]

    B = 4096
    for epoch in range(60):
        p = torch.randperm(X_tr.size(0), device=DEVICE)
        total = 0.0
        for i in range(0, X_tr.size(0), B):
            idx = p[i:i+B]
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            pred = model(xb).squeeze(-1)     
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        with torch.no_grad():
            val = lossf(model(X_va).squeeze(-1), y_va).item()

        if (epoch+1) % 5 == 0:
            print(f"[epoch {epoch+1:02d}] train={total/X_tr.size(0):.4f}  val={val:.4f}")

    torch.save(model.state_dict(), "artifacts/pm_stgnn.pt")
    joblib.dump(scaler, "artifacts/pm_scaler.pkl")
    print("[OK] saved artifacts/pm_stgnn.pt and artifacts/pm_scaler.pkl")

if __name__ == "__main__":
    main()
