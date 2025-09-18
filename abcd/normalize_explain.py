# filename: test_normalization.py
# 목적: 프로젝트의 "max-abs 정규화" 로직이 제대로 작동하는지 자동으로 검증한다.

import torch                    # 텐서 계산 라이브러리
from dataclasses import dataclass
from typing import List, Dict
import csv
import math

# ===== ① MiniGraph: PyG(Data) 대체용 작은 컨테이너 =====
# - 실제 프로젝트는 torch_geometric.data.Data 를 쓴다.
# - 여기선 검증 목적이라 필요한 속성(x, edge_attr, ground_motion, y)만 가진 단순 구조체를 쓴다.
@dataclass
class MiniGraph:
    x: torch.Tensor             # (N_nodes, 15) 노드 특성 15차원 (프로젝트 슬라이스 규약과 동일)
    edge_attr: torch.Tensor     # (N_edges, >= 6) 엣지 특성 (우리는 열 2,3,4,5만 사용)
    ground_motion: torch.Tensor # (T,) 지반가속 시계열
    y: torch.Tensor             # (M,) 회귀 타깃(예: 변위 등)

# ===== ② TestGraphDataset: 당신 코드와 동일한 정규화 메서드 구현 =====
class TestGraphDataset:
    def __init__(self, graphs: List[MiniGraph], response_type: str = "Acceleration"):
        self.graphs = graphs                     # 그래프 리스트
        self.response_type = response_type       # 응답 타입(정보용)
        # 아래 두 플래그는 "중복 정규화"를 막는다.
        self.source_normalization_state = False  # x/edge_attr/ground_motion 정규화 여부
        self.target_normalization_state = False  # y 정규화 여부

    # --- 정규화에 쓸 "스케일(분모)" 계산 ---
    # 데이터셋 전체(self.graphs)를 훑어 각 블록의 "전역 최대 절대값 max(|.|)"를 구한다.
    # 반환: {'x': {...}, 'edge_attr': {...}, 'ground_motion': tensor, 'y': tensor}
    def get_normalized_item_dict(self):
        normalized_item_dict = {}
        x = {}
        edge_attr = {}
        # 아직 한 번도 정규화를 안 했을 때(중복 계산 방지)
        if not self.source_normalization_state and not self.target_normalization_state:
            # x의 각 블록별 전역 max-abs. (프로젝트의 슬라이스와 1:1 대응)
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

            # edge_attr에서 실제 사용하는 열(2~5)의 전역 max-abs
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

            # 딕셔너리에 정리
            normalized_item_dict["x"] = x

            # 시계열/타깃도 전역 max-abs
            normalized_item_dict["ground_motion"] = max(
                [torch.max(torch.abs(data.ground_motion)) for data in self.graphs]
            )
            normalized_item_dict["y"] = max(
                [torch.max(torch.abs(data.y)) for data in self.graphs]
            )
            normalized_item_dict["edge_attr"] = edge_attr
            normalized_item_dict["response_type"] = self.response_type
        return normalized_item_dict

    # --- 입력(source) 정규화: x, edge_attr, ground_motion 나누기 ---
    def normalize_source(self, normalized_item_dict):
        if self.source_normalization_state is False:
            for data in self.graphs:
                # x의 각 블록을 해당 스케일로 나눈다.
                data.x[:, :3]     = data.x[:, :3]     / normalized_item_dict["x"]["XYZ_gridline_num"]
                data.x[:, 3:6]    = data.x[:, 3:6]    / normalized_item_dict["x"]["XYZ_grid_index"]
                data.x[:, 6]      = data.x[:, 6]      / normalized_item_dict["x"]["period"]
                data.x[:, 7]      = data.x[:, 7]      / normalized_item_dict["x"]["DOF"]
                data.x[:, 8]      = data.x[:, 8]      / normalized_item_dict["x"]["mass"]
                data.x[:, 9:12]   = data.x[:, 9:12]   / normalized_item_dict["x"]["XYZ_inertia"]
                data.x[:, 12:15]  = data.x[:, 12:15]  / normalized_item_dict["x"]["XYZ_mode_shape"]

                # edge_attr 4개 열
                data.edge_attr[:, 2] = data.edge_attr[:, 2] / normalized_item_dict["edge_attr"]["S_y"]
                data.edge_attr[:, 3] = data.edge_attr[:, 3] / normalized_item_dict["edge_attr"]["S_z"]
                data.edge_attr[:, 4] = data.edge_attr[:, 4] / normalized_item_dict["edge_attr"]["area"]
                data.edge_attr[:, 5] = data.edge_attr[:, 5] / normalized_item_dict["edge_attr"]["element_length"]

                # 지반가속 시계열
                data.ground_motion   = data.ground_motion   / normalized_item_dict["ground_motion"]

            # 중복 방지를 위한 플래그 on
            self.source_normalization_state = True

    # --- 타깃(target) 정규화: y 나누기 ---
    def normalize_target(self, normalized_item_dict):
        if self.target_normalization_state is False:
            for data in self.graphs:
                data.y = data.y / normalized_item_dict["y"]
            self.target_normalization_state = True

# ===== ③ 합성 그래프 만들기: 다양한 스케일을 가진 랜덤 데이터 =====
def build_synthetic_graph(
    n_nodes: int = 8,             # 노드 수
    n_edges: int = 10,            # 엣지 수
    gm_len: int = 500,            # ground_motion 길이(T)
    seed: int = 0,                # 재현성 위한 시드
    scales: Dict[str, float] = None  # 특정 블록의 스케일(최대 크기)을 덮어쓸 수 있음
) -> MiniGraph:
    g = torch.Generator().manual_seed(seed)  # torch 랜덤 제너레이터(시드 고정)

    # 각 블록별 대표 스케일(최댓값 크기 근사) - 0 방지/다양한 크기
    default_scales = dict(
        gridline=50.0,    # x[:, :3]
        gridindex=5.0,    # x[:, 3:6]
        period=2.5,       # x[:, 6] (양수)
        dof=6.0,          # x[:, 7] (정수 양수)
        mass=1000.0,      # x[:, 8] (양수)
        inertia=5e4,      # x[:, 9:12] (양수)
        modeshape=0.8,    # x[:, 12:15] (±)
        Sy=300.0,         # edge_attr[:, 2] (±)
        Sz=200.0,         # edge_attr[:, 3] (±)
        area=0.03,        # edge_attr[:, 4] (양수)
        elen=12.0,        # edge_attr[:, 5] (양수)
        gm=4.0,           # ground_motion (±)
        y=120.0           # 타깃 (±)
    )
    if scales:                          # 사용자가 특정 블록 스케일을 덮어쓰면 반영
        default_scales.update(scales)
    s = default_scales

    # ---- x (N, 15) 생성: 프로젝트 규약대로 각 슬라이스 분포 설정 ----
    x = torch.zeros((n_nodes, 15))

    # [:, :3]  gridline_num  → -gridline ~ +gridline
    x[:, :3]      = (torch.rand((n_nodes, 3), generator=g) * 2 - 1) * s["gridline"]
    # [:, 3:6]  grid_index    → -gridindex ~ +gridindex
    x[:, 3:6]     = (torch.rand((n_nodes, 3), generator=g) * 2 - 1) * s["gridindex"]
    # [:, 6]    period        → 0.1 ~ period (양수, 0 나눔 회피 위해 하한 clamp)
    x[:, 6]       = (torch.rand(n_nodes, generator=g) * s["period"]).clamp(min=0.1)
    # [:, 7]    DOF           → 1 ~ dof (정수), 학습 편의상 float로 보관
    x[:, 7]       = torch.randint(1, int(max(2, s["dof"])) + 1, (n_nodes,), generator=g).float()
    # [:, 8]    mass          → 1.0 ~ mass (양수, 0 회피)
    x[:, 8]       = (torch.rand(n_nodes, generator=g) * s["mass"]).clamp(min=1.0)
    # [:, 9:12] inertia       → 10.0 ~ inertia (양수, 0 회피)
    x[:, 9:12]    = (torch.rand((n_nodes, 3), generator=g) * s["inertia"]).clamp(min=10.0)
    # [:, 12:15] mode shape   → -modeshape ~ +modeshape (± 부호 보존)
    x[:, 12:15]   = (torch.rand((n_nodes, 3), generator=g) * 2 - 1) * s["modeshape"]

    # ---- edge_attr (E, 6) 생성: 우리가 쓰는 열만 채운다(2~5) ----
    edge_attr = torch.zeros((n_edges, 6))
    edge_attr[:, 2] = (torch.rand(n_edges, generator=g) * 2 - 1) * s["Sy"]
    edge_attr[:, 3] = (torch.rand(n_edges, generator=g) * 2 - 1) * s["Sz"]
    edge_attr[:, 4] = (torch.rand(n_edges, generator=g) * s["area"]).clamp(min=1e-4)
    edge_attr[:, 5] = (torch.rand(n_edges, generator=g) * s["elen"]).clamp(min=0.2)

    # ---- 시계열/타깃 ----
    ground_motion = (torch.rand(gm_len, generator=g) * 2 - 1) * s["gm"]  # ±
    y = (torch.rand(64, generator=g) * 2 - 1) * s["y"]                   # ±

    return MiniGraph(x=x, edge_attr=edge_attr, ground_motion=ground_motion, y=y)

# ===== ④ 전역 최대 절대값 계산 유틸 =====
# 데이터셋 전체 그래프를 훑어 각 블록별 max(|.|)를 딕셔너리로 돌려준다.
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

# ===== ⑤ 결과 표 출력 =====
def print_table(before: Dict[str, float], after: Dict[str, float], tol: float = 1e-6):
    # 헤더 출력(콘솔 표)
    header = f"{'feature_block':35s} | {'before_max_abs':>14s} | {'after_max_abs':>13s} | {'is_normalized(~=1)':>18s}"
    print(header)
    print("-"*len(header))
    all_ok = True
    rows = []
    for k in sorted(before.keys()):     # 블록 이름으로 정렬해서 보기 좋게
        b = before[k]                   # 정규화 전 전역 max-abs
        a = after[k]                    # 정규화 후 전역 max-abs
        ok = abs(a - 1.0) < tol         # 1.0과 근접한지(허용오차 tol)
        all_ok = all_ok and ok
        rows.append((k, b, a, ok))      # CSV 저장용
        print(f"{k:35s} | {b:14.6g} | {a:13.6g} | {str(ok):>18s}")
    print("-"*len(header))
    print("All blocks normalized to max-abs ~ 1.0:", all_ok)
    return rows, all_ok                 # 표 데이터, 전체 성공 여부 반환

# ===== ⑥ CSV 저장 =====
def save_csv(rows, path="normalization_check.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_block", "before_max_abs", "after_max_abs", "is_normalized(~=1)"])
        for k, b, a, ok in rows:
            w.writerow([k, b, a, ok])

# ===== ⑦ 메인 실행부 =====
def main():
    # (a) 서로 다른 스케일을 가진 그래프 3개 생성
    #     - 전역 max-abs가 진짜 "전역"을 잡는지 검증하기 위해 그래프마다 큰 값이 다른 블록에서 나오게 한다.
    g1 = build_synthetic_graph(seed=42)
    g2 = build_synthetic_graph(seed=7,   scales=dict(gridline=70.0, area=0.05))
    g3 = build_synthetic_graph(seed=123, scales=dict(mass=1500.0, Sy=500.0, gm=6.0, y=200.0))
    dataset = TestGraphDataset([g1, g2, g3], response_type="Acceleration")

    # (b) 정규화 "전" 전역 max-abs 스냅샷
    before = compute_global_max_abs(dataset.graphs)

    # (c) 스케일 계산 + 정규화 적용
    norm = dataset.get_normalized_item_dict()  # 전역 max-abs 사전
    dataset.normalize_source(norm)             # x/edge_attr/ground_motion 정규화
    dataset.normalize_target(norm)             # y 정규화

    # (d) 정규화 "후" 전역 max-abs 스냅샷
    after = compute_global_max_abs(dataset.graphs)

    # (e) 표 출력 + CSV 저장 + 실패 시 에러 처리
    rows, ok = print_table(before, after, tol=1e-6)
    save_csv(rows, "normalization_check.csv")
    if not ok:
        # 어떤 블록이라도 max-abs가 1과 충분히 가깝지 않으면 실패로 간주
        raise SystemExit("정규화 검증 실패: 일부 블록의 정규화 후 max-abs가 1.0과 충분히 가깝지 않습니다.")

if __name__ == "__main__":
    main()
