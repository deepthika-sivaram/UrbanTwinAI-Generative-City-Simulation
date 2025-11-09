import os
import numpy as np
import joblib
import torch
import torch.nn as nn

FEATS = ["building_cov", "green_cov", "road_den", "impervious", "traffic_level", "uhi_raw"]

class STGCN(nn.Module):

    def __init__(self, in_dim: int, hidden: int = 64, arch: str = "mlp3"):
        super().__init__()
        if arch == "mlp2":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else: 
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x, edge_index=None):
        return self.net(x).squeeze(-1)

class PMSTGNN:
    def __init__(self, model_path: str | None = None, scaler_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = STGCN(in_dim=len(FEATS), arch="mlp3").to(self.device)
        self.scaler = None
        self._ready = False

        if model_path and scaler_path and os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                ok = self._load_ckpt_forgiving(model_path)   
                if ok:
                    self.model.eval()
                    self.scaler = joblib.load(scaler_path)
                    self._ready = True
                    print("[PM] STGNN weights+scaler loaded")
                else:
                    print("[PM] Checkpoint load returned False; using fallback.")
            except Exception as e:
                print(f"[PM] Failed to load artifacts: {e}. Using fallback.")
        else:
            print("[PM] Artifacts not found. Using fallback heuristic model.")

    def is_ready(self) -> bool:
        return bool(self._ready)

    def backend_name(self) -> str:
        return "STGNN" if self._ready else "heuristic"

    def _fallback(self, df):
        return (
            0.6*df["impervious"]
            - 0.4*df["green_cov"]
            + 0.3*df["traffic_level"]
        ).to_numpy(dtype=np.float32)

    def predict_map(self, base_df, scen_df, adj=None, T=6, weather=None, traffic=None):
        Xb = base_df[FEATS].to_numpy(dtype=np.float32)
        Xs = scen_df[FEATS].to_numpy(dtype=np.float32)

        if self.is_ready() and self.scaler is not None:
            Xb = self.scaler.transform(Xb)
            Xs = self.scaler.transform(Xs)
            with torch.no_grad():
                yb = self.model(torch.from_numpy(Xb).to(self.device)).cpu().numpy()
                ys = self.model(torch.from_numpy(Xs).to(self.device)).cpu().numpy()
        else:
            yb = self._fallback(base_df)
            ys = self._fallback(scen_df)

        pm_map = ys - yb
        return pm_map, float(pm_map.mean())
    

    def _load_ckpt_forgiving(self, ckpt_path: str) -> bool:
        import torch

        sd = torch.load(ckpt_path, map_location=self.device)

        try:
            self.model.load_state_dict(sd, strict=True)
            return True
        except Exception:
            pass

        if any(k.split('.')[0].isdigit() for k in sd.keys()):
            sd = { (k if k.startswith("net.") else f"net.{k}"): v for k, v in sd.items() }

        try:
            self.model.load_state_dict(sd, strict=True)
            return True
        except Exception:
            pass

        def _shape(t):
            try:
                return tuple(t.shape)
            except Exception:
                return None

        has_64_to_1 = (
            ("net.2.weight" in sd and _shape(sd["net.2.weight"]) == (1, 64)) or
            ("2.weight" in sd and _shape(sd["2.weight"]) == (1, 64))
        )

        if has_64_to_1:
            in_dim = self.model.net[0].in_features 
            self.model = STGCN(in_dim=in_dim, arch="mlp2").to(self.device)
            try:
                self.model.load_state_dict(sd, strict=True)
                return True
            except Exception:
                self.model.load_state_dict(sd, strict=False)
                return True

        self.model.load_state_dict(sd, strict=False)
        return True
