"""训练、推理和性能测试共享的设备与检查点工具。"""

from pathlib import Path
import torch

from .config import ModelConfig
from .tokenizer import load_tokenizer
from .model import DecoderLM


def choose_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable')
    return device


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = DecoderLM(ModelConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    tokenizer = load_tokenizer(checkpoint['tokenizer'])
    return model, tokenizer, checkpoint
