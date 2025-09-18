# filename: test_normalization.py
# 목적: 사용 중인 정규화 로직(max-abs 스케일링)이 의도대로 동작하는지 자동 검증

import torch
from dataclasses import dataclass
from typing import List, Dict
import csv
import math

# ----- Mini Graph (torch_geometric.data.Data 대체용) -----
@dataclass
class MiniGraph:
    x: torch.Tensor             # (N_nodes, 15)
    edge_attr: torch.Tensor     # (N_edges, >= 6) -> cols [2,3,4,5] 사용
    ground_motion: torch.Tensor # (T,)
    y: torch.Tensor             # (M,)

# ----- 사용자 코드 스펙을 그대로 반영한 Dataset -----
class TestGraphDataset:
    def __init__(self, graphs: List[MiniGraph], response_type: str = "Acceleration"):
        self.graphs = graphs
        self.response_type = response_type
        self.source_normalization_state = False
        self.target_normalization_state = False

    def get_normalized_item_dict(self):
        normalized_item_dict = {}
        x = {}
        edge_attr = {}
        if not self.source_normalization_state and not self.target_normalization_state:
            x["XYZ_gridline_num"] = max(
                [torch.max(torch.abs(data.x[:, :3])) for data in self.graphs]
            )
            x["XYZ_grid_index"] = max(
                [torch.max(torch.abs(data.x[:, 3:6])) for data in self.graphs]
            )
            x["period"] = max(
                [torch.max(torch.abs(data.x[:, 6])) for data in self.graphs]
            )
            x["DOF"] = max([torch.max(torch.abs(data.x[:, 7])) for data in self.graphs])
            x["mass"] = max(
                [torch.max(torch.abs(data.x[:, 8])) for data in self.graphs]
            )
            x["XYZ_inertia"] = max(
                [torch.max(torch.abs(data.x[:, 9:12])) for data in self.graphs]
            )
            x["XYZ_mode_shape"] = max(
                [torch.max(torch.abs(data.x[:, 12:15])) for data in self.graphs]
            )
            edge_attr["S_y"] = max(
                [torch.max(torch.abs(data.edge_attr[:, 2])) for data in self.graphs]
            )
            edge_attr["S_z"] = max(
                [torch.max(torch.abs(data.edge_attr[:, 3])) for data in self.graphs]
            )
            edge_attr["area"] = max(
                [torch.max(torch.abs(data.edge_attr[:, 4])) for data in self.graphs]
            )
            edge_attr["element_length"] = max(
                [torch.max(torch.abs(data.edge_attr[:, 5])) for data in self.graphs]
            )
            normalized_item_dict["x"] = x
            normalized_item_dict["ground_motion"] = max(
                [torch.max(torch.abs(data.ground_motion)) for data in self.graphs]
            )
            normalized_item_dict["y"] = max(
                [torch.max(torch.abs(data.y)) for data in self.graphs]
            )
            normalized_item_dict["edge_attr"] = edge_attr
            normalized_item_dict["response_type"] = self.response_type
        return normalized_item_dict

    def normalize_source(self, normalized_item_dict):
        if self.source_normalization_state is False:
            for data in self.graphs:
                data.x[:, :3]     = data.x[:, :3]     / normalized_item_dict["x"]["XYZ_gridline_num"]
                data.x[:, 3:6]    = data.x[:, 3:6]    / normalized_item_dict["x"]["XYZ_grid_index"]
                data.x[:, 6]      = data.x[:, 6]      / normalized_item_dict["x"]["period"]
                data.x[:, 7]      = data.x[:, 7]      / normalized_item_dict["x"]["DOF"]
                data.x[:, 8]      = data.x[:, 8]      / normalized_item_dict["x"]["mass"]
                data.x[:, 9:12]   = data.x[:, 9:12]   / normalized_item_dict["x"]["XYZ_inertia"]
                data.x[:, 12:15]  = data.x[:, 12:15]  / normalized_item_dict["x"]["XYZ_mode_shape"]
                data.edge_attr[:, 2] = data.edge_attr[:, 2] / normalized_item_dict["edge_attr"]["S_y"]
                data.edge_attr[:, 3] = data.edge_attr[:, 3] / normalized_item_dict["edge_attr"]["S_z"]
                data.edge_attr[:, 4] = data.edge_attr[:, 4] / normalized_item_dict["edge_attr"]["area"]
                data.edge_attr[:, 5] = data.edge_attr[:, 5] / normalized_item_dict["edge_attr"]["element_length"]
                data.ground_motion   = data.ground_motion   / normalized_item_dict["ground_motion"]
            self.source_normalization_state = True

    def normalize_target(self, normalized_item_dict):
        if self.target_normalization_state is False:
            for data in self.graphs:
                data.y = data.y / normalized_item_dict["y"]
            self.target_normalization_state = True

# ----- 합성 그래프 생성 (스케일을 다르게 줘 전역 max-abs 테스트) -----
def build_synthetic_graph(
    n_nodes: int = 8, n_edges: int = 10, gm_len: int = 500, seed: int = 0,
    scales: Dict[str, float] = None
) -> MiniGraph:
    g = torch.Generator().manual_seed(seed)
    # 기본 스케일 (0 방지 & 다양하게)
    default_scales = dict(
        gridline=50.0, gridindex=5.0, period=2.5, dof=6.0, mass=1000.0,
        inertia=5e4, modeshape=0.8, Sy=300.0, Sz=200.0, area=0.03, elen=12.0,
        gm=4.0, y=120.0
    )
    if scales:
        default_scales.update(scales)
    s = default_scales

    # x: (N, 15)
    x = torch.zeros((n_nodes, 15))
    x[:, :3]      = (torch.rand((n_nodes, 3), generator=g) * 2 - 1) * s["gridline"]    # +/- 
    x[:, 3:6]     = (torch.rand((n_nodes, 3), generator=g) * 2 - 1) * s["gridindex"]   # +/- 
    x[:, 6]       = (torch.rand(n_nodes, generator=g) * s["period"]).clamp(min=0.1)    # +
    x[:, 7]       = torch.randint(1, int(max(2, s["dof"])) + 1, (n_nodes,), generator=g).float() # +
    x[:, 8]       = (torch.rand(n_nodes, generator=g) * s["mass"]).clamp(min=1.0)      # +
    x[:, 9:12]    = (torch.rand((n_nodes, 3), generator=g) * s["inertia"]).clamp(min=10.0) # +
    x[:, 12:15]   = (torch.rand((n_nodes, 3), generator=g) * 2 - 1) * s["modeshape"]   # +/-

    # edge_attr: 6 cols, [2..5] 사용
    edge_attr = torch.zeros((n_edges, 6))
    edge_attr[:, 2] = (torch.rand(n_edges, generator=g) * 2 - 1) * s["Sy"]            # +/-
    edge_attr[:, 3] = (torch.rand(n_edges, generator=g) * 2 - 1) * s["Sz"]            # +/-
    edge_attr[:, 4] = (torch.rand(n_edges, generator=g) * s["area"]).clamp(min=1e-4)  # +
    edge_attr[:, 5] = (torch.rand(n_edges, generator=g) * s["elen"]).clamp(min=0.2)   # +

    # ground_motion: +/- 
    ground_motion = (torch.rand(gm_len, generator=g) * 2 - 1) * s["gm"]

    # target y: +/-
    y = (torch.rand(64, generator=g) * 2 - 1) * s["y"]

    return MiniGraph(x=x, edge_attr=edge_attr, ground_motion=ground_motion, y=y)

# ----- 전역 max(|.|) 계산 -----
def compute_global_max_abs(graphs: List[MiniGraph]) -> Dict[str, float]:
    out = {}
    out["x[:,:3]_gridline_num"] = max([torch.max(torch.abs(G.x[:, :3])).item() for G in graphs])
    out["x[:,3:6]_grid_index"]  = max([torch.max(torch.abs(G.x[:, 3:6])).item() for G in graphs])
    out["x[:,6]_period"]        = max([torch.max(torch.abs(G.x[:, 6])).item() for G in graphs])
    out["x[:,7]_DOF"]           = max([torch.max(torch.abs(G.x[:, 7])).item() for G in graphs])
    out["x[:,8]_mass"]          = max([torch.max(torch.abs(G.x[:, 8])).item() for G in graphs])
    out["x[:,9:12]_inertia"]    = max([torch.max(torch.abs(G.x[:, 9:12])).item() for G in graphs])
    out["x[:,12:15]_modeShape"] = max([torch.max(torch.abs(G.x[:, 12:15])).item() for G in graphs])
    out["edge_attr[:,2]_Sy"]    = max([torch.max(torch.abs(G.edge_attr[:, 2])).item() for G in graphs])
    out["edge_attr[:,3]_Sz"]    = max([torch.max(torch.abs(G.edge_attr[:, 3])).item() for G in graphs])
    out["edge_attr[:,4]_area"]  = max([torch.max(torch.abs(G.edge_attr[:, 4])).item() for G in graphs])
    out["edge_attr[:,5]_elen"]  = max([torch.max(torch.abs(G.edge_attr[:, 5])).item() for G in graphs])
    out["ground_motion"]        = max([torch.max(torch.abs(G.ground_motion)).item() for G in graphs])
    out["y"]                    = max([torch.max(torch.abs(G.y)).item() for G in graphs])
    return out

def print_table(before: Dict[str, float], after: Dict[str, float], tol: float = 1e-6):
    header = f"{'feature_block':35s} | {'before_max_abs':>14s} | {'after_max_abs':>13s} | {'is_normalized(~=1)':>18s}"
    print(header)
    print("-"*len(header))
    all_ok = True
    rows = []
    for k in sorted(before.keys()):
        b = before[k]
        a = after[k]
        ok = abs(a - 1.0) < tol
        all_ok = all_ok and ok
        rows.append((k, b, a, ok))
        print(f"{k:35s} | {b:14.6g} | {a:13.6g} | {str(ok):>18s}")
    print("-"*len(header))
    print("All blocks normalized to max-abs ~ 1.0:", all_ok)
    return rows, all_ok

def save_csv(rows, path="normalization_check.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_block", "before_max_abs", "after_max_abs", "is_normalized(~=1)"])
        for k, b, a, ok in rows:
            w.writerow([k, b, a, ok])

def main():
    # 1) 서로 다른 스케일을 가진 그래프 3개 생성(전역 max-abs 검증에 유리)
    g1 = build_synthetic_graph(seed=42)
    g2 = build_synthetic_graph(seed=7,   scales=dict(gridline=70.0, area=0.05))
    g3 = build_synthetic_graph(seed=123, scales=dict(mass=1500.0, Sy=500.0, gm=6.0, y=200.0))
    dataset = TestGraphDataset([g1, g2, g3], response_type="Acceleration")

    # 2) 정규화 전 전역 max-abs
    before = compute_global_max_abs(dataset.graphs)

    # 3) 스케일 산출 + 정규화 적용
    norm = dataset.get_normalized_item_dict()
    dataset.normalize_source(norm)
    dataset.normalize_target(norm)

    # 4) 정규화 후 전역 max-abs
    after = compute_global_max_abs(dataset.graphs)

    # 5) 표 출력 & CSV 저장 & 어서션
    rows, ok = print_table(before, after, tol=1e-6)
    save_csv(rows, "normalization_check.csv")

    # 실패 시 명확히 알림
    if not ok:
        raise SystemExit("정규화 검증 실패: 일부 블록의 정규화 후 max-abs가 1.0과 충분히 가깝지 않습니다.")

if __name__ == "__main__":
    main()
