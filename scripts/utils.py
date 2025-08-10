import numpy as np, random, os
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_npz(path, X, ids=None):
    import numpy as np
    np.savez_compressed(path, X=X, ids=ids if ids is not None else np.arange(X.shape[0]))

def load_npz(path):
    import numpy as np
    d = np.load(path)
    return d["X"], d["ids"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
