import copy
import torch
import torch.nn as nn
from models import Autoformer as AutoformerBackbone

class Model(nn.Module):
    """
    AutoformerProjected:
    Wraps a pretrained Autoformer teacher (e.g., 321 features)
    with adapters to fine-tune on any dataset with a different feature count.
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        assert hasattr(configs, "teacher_ckpt"), "Missing --teacher_ckpt argument"
        assert hasattr(configs, "teacher_dim"), "Missing --teacher_dim argument"

        self.input_dim = configs.enc_in        # number of features in new dataset
        self.teacher_dim = configs.teacher_dim # dimension teacher was trained on (e.g., 321)

        # Projection adapters
        self.proj_in = nn.Linear(self.input_dim, self.teacher_dim)
        self.proj_out = nn.Linear(self.teacher_dim, self.input_dim)

        # Create a teacher backbone using teacher_dim
        tcfg = copy.deepcopy(configs)
        tcfg.enc_in = tcfg.dec_in = tcfg.c_out = self.teacher_dim
        self.teacher = AutoformerBackbone.Model(tcfg)

        # Load pretrained teacher weights
        ckpt = torch.load(configs.teacher_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        missing, unexpected = self.teacher.load_state_dict(state, strict=False)
        print(f"[AutoformerProjected] Loaded teacher. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        # Freeze/unfreeze
        freeze_flag = getattr(configs, "freeze_teacher", 1)
        for p in self.teacher.parameters():
            p.requires_grad = not bool(int(freeze_flag))
        print(f"[AutoformerProjected] Backbone frozen = {bool(int(freeze_flag))}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Project new dataset inputs to teacher space
        x_enc = self.proj_in(x_enc)
        x_dec = self.proj_in(x_dec)

        y = self.teacher(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(y, tuple):  # if attention weights returned
            y = y[0]
        y = self.proj_out(y)
        return y
