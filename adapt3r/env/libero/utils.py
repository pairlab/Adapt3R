import os
from hydra.utils import to_absolute_path
from tqdm import trange
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
np.set_printoptions(suppress=True)

from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from transformers import AutoModel, AutoTokenizer, logging

from libero.libero.benchmark import get_benchmark

import adapt3r.utils.file_utils as FileUtils
import adapt3r.utils.obs_utils as ObsUtils
from adapt3r.utils.dataset import SequenceDataset


BOUNDARIES_TIGHT = {
    'KITCHEN': (
        (-.5, -.6, 0),
        ( .5,  .6, 2.0)
    ),
    'LIVING_ROOM': (
        (-.3, -0.5, 0),
        (0.5,  0.5, 2)
    ),
    'STUDY': (
        (-.5, -.5, 0),
        ( .5,  .5, 2)
    ),
}
BOUNDARIES = {
    'KITCHEN': (
        (-1, -1, 0),
        ( 1,  1, 2.0)
    ),
    'LIVING_ROOM': (
        (-1, -1, 0),
        (1,  1, 2)
    ),
    'STUDY': (
        (-1, -1, 0),
        ( 1,  1, 2)
    ),
    'KITCHEN_DISTRACTOR': (
        (-1, -1, 0),
        ( 1,  1, 2.0)
    ),
    'LIVING_ROOM_DISTRACTOR': (
        (-1, -1, 0),
        (1,  1, 2)
    ),
    'STUDY_DISTRACTOR': (
        (-1, -1, 0),
        ( 1,  1, 2)
    ),
}



def get_benchmark_instance(benchmark_name, distractor=False, robot='Panda'):
    if distractor:
        benchmark_name = f'{benchmark_name}_distractor'
    if robot != 'Panda':
        benchmark_name = f'{benchmark_name}_{robot}'
    benchmark = get_benchmark(benchmark_name)()
    return benchmark
    

def get_boundaries(benchmark_name, tight=False):
    benchmark = get_benchmark(benchmark_name)()
    task_names = benchmark.get_task_names()
    boundaries = []
    for task_name in task_names:
        setting, _, _ = deconstruct_task_name(task_name)
        if tight:
            boundaries.append(BOUNDARIES_TIGHT[setting])
        else:
            boundaries.append(BOUNDARIES[setting])
    boundaries = np.array(boundaries)
    return boundaries



def deconstruct_task_name(task_name):
    scene_idx = task_name.find('SCENE')
    underscore_idx = task_name[scene_idx:].find('_')
    scene_name_len = scene_idx + underscore_idx
    setting = task_name[:scene_idx - 1]
    number = int(task_name[scene_idx+5:scene_name_len])
    instruction = task_name[scene_name_len+1:]
    return setting, number, instruction


def build_dataset(data_prefix,
                  suite_name,
                  benchmark_name, 
                  mode, 
                  seq_len, 
                  frame_stack,
                  shape_meta,
                  n_demos,
                  hdf5_cache_mode='low_dim',
                  extra_obs_modality=None,
                  obs_seq_len=1, 
                  load_obs=True,
                  task_embedding_format="clip",
                  load_next_obs=False,
                  dataset_keys=('actions',),
                  ):
    benchmark = get_benchmark(benchmark_name)()
    n_tasks = benchmark.n_tasks
    few_shot_demos = [1, 5, 10, 20, 45] if mode == 'fewshot' else None
    few_shot_demos_list = [f"demo_{i}" for i in few_shot_demos] if few_shot_demos is not None else None
    
    manip_datasets = []
    descriptions = []
    obs_modality = {
        'rgb': list(shape_meta['observation']['rgb'].keys()),
        'depth': list(shape_meta['observation']['depth'].keys()),
        'low_dim': list(shape_meta['observation']['lowdim'].keys()) + list(shape_meta['observation']['pointcloud'].keys())
    }
    if extra_obs_modality is not None:
        for key in extra_obs_modality:
            obs_modality[key] = obs_modality[key] + extra_obs_modality[key]
    
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    for i in trange(n_tasks):
        task_i_dataset = get_dataset(
            dataset_path=os.path.join(
                data_prefix, suite_name, benchmark.get_task_demonstration(i)
            ),
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            frame_stack=frame_stack,
            load_obs=load_obs,
            few_demos = few_shot_demos_list,
            n_demos=n_demos,
            hdf5_cache_mode=hdf5_cache_mode,
            load_next_obs=load_next_obs,
            dataset_keys=dataset_keys
        )
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
    task_embs = get_task_embs(task_embedding_format, descriptions)
    benchmark.set_task_embs(task_embs)
    datasets = [
        SequenceVLDataset(ds, emb, i) for i,(ds, emb) in enumerate(zip(manip_datasets, task_embs))
    ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")
    return concat_dataset

def get_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    load_obs=True,
    few_demos=None,
    n_demos=None,
    load_next_obs=False,
    dataset_keys=None,
    ):
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    seq_len = seq_len
    filter_key = filter_key
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []
    
    if dataset_keys is None:
        dataset_keys = ['actions',]
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        load_next_obs=load_next_obs,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        obs_seq_length=obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        few_demos=few_demos,
        n_demos=n_demos,
    )
    return dataset

class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb, task_id):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return_dict["task_id"] = self.task_id
        return return_dict

def get_task_embs(task_embedding_format, descriptions):
    logging.set_verbosity_error()
    if task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    return task_embs
