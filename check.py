import torch
from pprint import pprint

def inspect_structure_graph(pt_path: str):
    # 안전 로딩: PyG Data를 화이트리스트에 등록
    try:
        from torch_geometric.data import Data
        torch.serialization.add_safe_globals([Data])
        g = torch.load(pt_path, map_location="cpu")  # weights_only=True (기본)
    except Exception as e1:
        print(f"[safe load failed] {e1}\n-> retry with weights_only=False (trusted file only)")
        g = torch.load(pt_path, map_location="cpu", weights_only=False)

    print(f"== file: {pt_path}")

    # Data 또는 dict 모두 지원, keys가 메서드/프로퍼티 어떤 경우든 안전 처리
    if hasattr(g, "keys"):
        kobj = getattr(g, "keys")
        keys = list(kobj()) if callable(kobj) else list(kobj)   # ★ 여기 포인트
        getter = lambda k: getattr(g, k)
    elif isinstance(g, dict):
        keys = list(g.keys())
        getter = lambda k: g[k]
    else:
        raise TypeError(f"Unsupported type: {type(g)}")

    report = {}
    for k in sorted(keys):
        v = getter(k)
        item = {"type": type(v).__name__}
        if torch.is_tensor(v):
            item["shape"] = list(v.shape)
            try:
                item["min"] = float(v.min())
                item["max"] = float(v.max())
            except Exception:
                pass
        elif isinstance(v, (int, float, str)):
            item["value"] = v
        report[k] = item

    pprint(report)
    return g


# 예시 경로
inspect_structure_graph("./Data/Nonlinear_Analysis/eval/ChiChi_DBE/structure_204/structure_graph.pt")
