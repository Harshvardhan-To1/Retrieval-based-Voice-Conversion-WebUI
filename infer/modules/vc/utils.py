import os

from fairseq import checkpoint_utils
from pathlib import Path
_CURR_DIR = Path(__file__).parent
SECOND = _CURR_DIR.parents[2] / "logs"


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(SECOND, topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )

CONFIG_PATH = _CURR_DIR / "hubert_base.pt"
def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(CONFIG_PATH)],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
