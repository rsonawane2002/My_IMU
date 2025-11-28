import json, os, numpy as np
def save_sequence(out_dir: str, idx: int, arrays: dict, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f'seq_{idx:05d}')
    for k,v in arrays.items():
        np.save(base + f'_{k}.npy', v)
    with open(base + '_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return base
