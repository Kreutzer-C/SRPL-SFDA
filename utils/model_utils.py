# -*- coding: utf-8 -*-
"""
Utility for loading source models from either SRPL-SFDA (PyMIC UNet)
or MemProp-SFDA (standard UNet) checkpoint formats.
"""
import os
import sys
import torch

# cuDNN may fail to initialize in some environments; fall back to native CUDA kernels.
torch.backends.cudnn.enabled = False


def _get_state_dict(ckpt_path: str, device: str = "cuda:0") -> dict:
    """Load checkpoint and unwrap MemProp-style wrapper if present."""
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt.get("model_state_dict", ckpt)


def _detect_arch(state_dict: dict) -> str:
    """
    Detect which UNet architecture a state_dict belongs to.

    MemProp  keys start with: inc., down1., up1., outc.
    SRPL     keys start with: encoder., decoder.
    """
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("encoder.") or first_key.startswith("decoder."):
        return "srpl"
    if first_key.startswith(("inc.", "down", "up", "outc.")):
        return "memprop"
    raise ValueError(f"Cannot detect UNet architecture from key: {first_key!r}")


def build_model_from_checkpoint(
    ckpt_path: str,
    num_classes: int,
    memprop_dir: str = "/workspace/MemProp-SFDA",
    unet_config: str = None,
    device: str = "cuda:0",
):
    """
    Auto-detect the checkpoint architecture and build + load the matching model.

    Args:
        ckpt_path:   Path to checkpoint (.pth).
        num_classes: Number of output classes.
        memprop_dir: Root directory of MemProp-SFDA (needed when arch=memprop).
        unet_config: Path to MemProp UNet JSON config. Defaults to
                     {memprop_dir}/networks/unet_config.json.
        device:      Torch device string.

    Returns:
        net: Loaded model on `device`, in eval mode.
    """
    state_dict = _get_state_dict(ckpt_path, device)
    arch = _detect_arch(state_dict)

    if arch == "srpl":
        # SRPL-SFDA's own PyMIC-based UNet
        srpl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if srpl_root not in sys.path:
            sys.path.insert(0, srpl_root)
        from networks.net_factory import net_factory
        net = net_factory("unet2d", in_chns=1, class_num=num_classes)

    elif arch == "memprop":
        # MemProp-SFDA's standard UNet (inc/down/up/outc)
        # Use importlib to avoid 'utils' namespace collision between the two projects.
        import importlib.util as ilu
        import json

        if unet_config is None:
            unet_config = os.path.join(memprop_dir, "networks", "unet_config.json")
        assert os.path.exists(unet_config), (
            f"MemProp UNet config not found: {unet_config}\n"
            "Pass --memprop_dir or --unet_config explicitly."
        )

        # Load simple_tools first (needed by unet_modeling)
        st_spec = ilu.spec_from_file_location(
            "_memprop_simple_tools",
            os.path.join(memprop_dir, "utils", "simple_tools.py"),
        )
        st_mod = ilu.module_from_spec(st_spec)
        st_spec.loader.exec_module(st_mod)
        sys.modules.setdefault("utils.simple_tools", st_mod)

        # Load unet_modeling
        um_spec = ilu.spec_from_file_location(
            "_memprop_unet_modeling",
            os.path.join(memprop_dir, "networks", "unet_modeling.py"),
        )
        um_mod = ilu.module_from_spec(um_spec)
        um_spec.loader.exec_module(um_mod)

        # Read JSON config and build model
        with open(unet_config) as f:
            cfg = json.load(f)
        net = um_mod.UNet(
            n_channels=cfg.get("in_channels", 1),
            n_classes=num_classes,
            first_channels=cfg.get("first_channels", 64),
            only_feature=cfg.get("only_feature", False),
            only_logits=cfg.get("only_logits", True),
            bilinear=cfg.get("bilinear", False),
        ).to(device)

    else:
        raise RuntimeError(f"Unknown architecture: {arch}")

    net.load_state_dict(state_dict)
    net.eval()
    return net
