import torch
import random
from torch.utils.data import Dataset
from typing import Optional, Union, Dict, Any, List


class SequenceVLDataset(Dataset):
    """Dataset wrapper that adds vision-language task information to sequence data.
    
    Args:
        sequence_dataset: Base sequence dataset
        task_emb: Optional task embedding tensor or list of tensors
        lang_inst: Optional language instruction string or list of strings
        task_id: Optional task ID (int/tensor) or list of task IDs
    """
    def __init__(
        self, 
        sequence_dataset: Dataset,
        task_emb: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        lang_inst: Optional[Union[str, List[str]]] = None,
        task_id: Optional[Union[int, torch.Tensor, List[Union[int, torch.Tensor]]]] = None
    ):
        self.sequence_dataset = sequence_dataset
        # Convert all inputs to lists
        self.task_emb = [task_emb] if task_emb is not None and not isinstance(task_emb, list) else task_emb
        self.lang_inst = [lang_inst] if lang_inst is not None and not isinstance(lang_inst, list) else lang_inst
        self.task_id = [task_id] if task_id is not None and not isinstance(task_id, list) else task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return_dict = self.sequence_dataset.__getitem__(idx)
        
        # Add any provided task information to the return dict, sampling from lists
        if self.task_emb is not None:
            return_dict["task_emb"] = random.choice(self.task_emb)
        if self.lang_inst is not None:
            return_dict["lang_inst"] = random.choice(self.lang_inst)
        if self.task_id is not None:
            return_dict["task_id"] = random.choice(self.task_id)
            
        return return_dict