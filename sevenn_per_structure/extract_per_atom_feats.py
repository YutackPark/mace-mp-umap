import sys
import tqdm
import torch

from torch.utils.data.dataloader import DataLoader

import ase.io as io
from ase.atoms import Atoms

from torch_geometric.loader.dataloader import Collater

from sevenn.train.dataload import atoms_to_graph
import sevenn._keys as KEY
import sevenn.util as util
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.atom_graph_data import AtomGraphData

from typing import List, Optional, Sequence, Any, Dict


"""
Save per atom features of dataset using 7net-0
"""

device = torch.device("cuda")

"""
cp = "/home/parkyutack/shared_data/pretrained/7net_chgTot/checkpoint_600.pth"
db = "/home/parkyutack/share/data/MPTrj2022/mptrj_test_dataset_uncorrected.extxyz"
out_path = f"./outputs/{db}.pth"
atoms_list = io.read(db, index=":", format='extxyz')
"""

cp = "/user/hansw/shared_data/pretrained/7net_chgTot/checkpoint_600.pth"

db = sys.argv[1]
basename = db.split("/")[-1].split(".")[0]
out_path = f"./outputs/{basename}.pth"
atoms_list = io.read(db, index=":", format='extxyz')


class AtomsToGraphCollater(Collater):

    def __init__(
        self,
        dataset: Sequence[Atoms],
        cutoff: float,
        type_map: Dict[int, int],  # Z -> node onehot
        requires_grad_key: str = KEY.POS,
        key_x: str = KEY.NODE_FEATURE,
        transfer_info: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        # quite original collator's type mismatch with []
        super().__init__([], follow_batch, exclude_keys)
        self.dataset = dataset
        self.cutoff = cutoff
        self.type_map = type_map
        self.requires_grad_key = requires_grad_key
        self.transfer_info = transfer_info
        self.key_x = key_x

    def _Z_to_onehot(self, Z):
        return torch.LongTensor(
            [self.type_map[z.item()] for z in Z]
        )

    def __call__(self, batch: List[Any]) -> Any:
        # build list of graph
        graph_list = []
        for stct in batch:
            graph = atoms_to_graph(
                stct, self.cutoff, transfer_info=self.transfer_info
            )
            graph = AtomGraphData.from_numpy_dict(graph)
            graph[self.key_x] = self._Z_to_onehot(graph[self.key_x])
            graph[self.requires_grad_key].requires_grad_(True)
            graph_list.append(graph)
        return super().__call__(graph_list)



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


def post(out):
    data_list = util.to_atom_graph_list(out)
    atomic_features_list = [
        out.atomic_features[i: i+n] 
        for i, n 
        in zip(out.ptr[:-1], out.num_atoms)
    ]
    for datum, af in zip(data_list, atomic_features_list):
        datum["atomic_features"] = af
        datum["task_id"] = datum.data_info["task_id"]
        datum["calc_id"] = datum.data_info["calc_id"]
        datum["ionic_step"] = datum.data_info["ionic_step"]
    _keys_to_save = [
        KEY.ENERGY, 
        KEY.PRED_TOTAL_ENERGY, 
        KEY.ATOMIC_ENERGY, 
        KEY.ATOMIC_NUMBERS,
        "task_id",
        "calc_id",
        "ionic_step",
        "atomic_features"
    ]
    ret = []
    for data in data_list:
        dct = {}
        for k, v in data.items():
            if k not in _keys_to_save:
                continue
            if isinstance(v, torch.Tensor):
                dct[k] = v.tolist()
            else:
                dct[k] = v
        ret.append(dct)
    return ret
                 
model, config = util.model_from_checkpoint(cp)

_layers = list(model._modules.items())

dump_x = DumpX(KEY.NODE_FEATURE, "atomic_features")
_layers = insert_after(
    "4_equivariant_gate", 
    ("dump_x", dump_x), 
    _layers,
)
model_patched = AtomGraphSequential(_layers, model.cutoff, model.type_map)


collate_fn = AtomsToGraphCollater(
        atoms_list, 
        model_patched.cutoff, 
        model_patched.type_map,
        transfer_info=True
    )

dataloader = DataLoader(
        atoms_list, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=collate_fn
    )

model_patched.to(device)

to_save = []
for batch in tqdm.tqdm(dataloader):
    batch.to(device)
    out = model_patched(batch)
    to_save.extend(post(out))

torch.save(to_save, out_path)

