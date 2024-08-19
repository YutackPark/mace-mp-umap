import torch

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.sevennet_calculator import SevenNetCalculator
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.atom_graph_data import AtomGraphData


class DumpX(torch.nn.Module):

    def __init__(self, data_key_x, data_key_out):
        super().__init__()
        self.data_key_x = data_key_x
        self.data_key_out = data_key_out

    def forward(self, data):
        data[self.data_key_out] = data[self.data_key_x].clone()
        return data


def insert_after(module_name_after, key_module_pair, layers):
    idx = -1
    for i, (key, _) in enumerate(layers):
        if key == module_name_after:
            idx = i
            break
    if idx == -1:
        assert False
    layers.insert(idx + 1, key_module_pair)
    return layers


def patch_calc_for_descriptor(
    calc: SevenNetCalculator, 
    insert_after_key: str = "4_equivariant_gate"
):
    model = calc.model
    _layers = list(model._modules.items())

    dump_x = DumpX(KEY.NODE_FEATURE, "atomic_features")
    _layers = insert_after(
        insert_after_key, 
        ("dump_x", dump_x), 
        _layers,
    )
    model_patched = AtomGraphSequential(_layers, model.cutoff, model.type_map)
    calc.model = model_patched
    return calc
