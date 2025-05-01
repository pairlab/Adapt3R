import os
import hydra
from omegaconf import OmegaConf

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark
from libero.libero.utils.task_generation_utils import get_task_info, \
    get_suite_generator_func, register_task_info
from libero.libero.utils.mu_utils import get_scene_class
from libero.libero.utils.bddl_generation_utils import *

import adapt3r.env.libero.utils as lu


OmegaConf.register_new_resolver("eval", eval, replace=True)
    
def generate_bddl_from_task_info(folder="tmp/bddl"):
    results = []
    failures = []
    bddl_file_names = []
    os.makedirs(folder, exist_ok=True)

    registered_task_info_dict = get_task_info()
    for scene_name in registered_task_info_dict:
        for task_info_tuple in registered_task_info_dict[scene_name]:
            scene_name = task_info_tuple.scene_name
            language = task_info_tuple.language
            objects_of_interest = task_info_tuple.objects_of_interest
            goal_states = task_info_tuple.goal_states
            scene = get_scene_class(scene_name)()

            result = get_suite_generator_func(scene.workspace_name)(
                language=language,
                xy_region_kwargs_list=scene.xy_region_kwargs_list,
                affordance_region_kwargs_list=scene.affordance_region_kwargs_list,
                fixture_object_dict=scene.fixture_object_dict,
                movable_object_dict=scene.movable_object_dict,
                objects_of_interest=objects_of_interest,
                init_states=scene.init_states,
                goal_states=goal_states,
            )
            result = get_result(result)
            bddl_file_name = save_to_file(
                result, scene_name=scene_name, language=language, folder=folder
            )
            if bddl_file_name in bddl_file_names:
                print(bddl_file_name)
            bddl_file_names.append(bddl_file_name)
            results.append(result)
    print(f"Succefully generated: {len(results)}")
    return bddl_file_names, failures

@hydra.main(config_path="../../config", 
            config_name='train_debug', 
            version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)


    benchmark_dict = benchmark.get_benchmark_dict()

    benchmark_name = cfg.task.benchmark_name
    # process each task
    benchmark_instance = benchmark_dict[benchmark_name]()
    task_names = benchmark_instance.get_task_names()
    for task_no, task_name in enumerate(task_names):
        bddl_file = benchmark_instance.get_task_bddl_file_path(task_no)
        parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file)
        setting, number, instruction = lu.deconstruct_task_name(task_name)
        scene_name = f'{setting.lower()}_distractor_scene{number}'
        goal_states = [tuple(goal_state) for goal_state in parsed_problem['goal_state']]
        register_task_info(
            language=' '.join(parsed_problem['language_instruction']),
            scene_name=scene_name,
            objects_of_interest=parsed_problem['obj_of_interest'],
            goal_states=goal_states
        )


    bddl_files, failures = generate_bddl_from_task_info(folder=f"tmp/bddl/{cfg.task.benchmark_name}")
    print(bddl_files)


if __name__ == "__main__":
    main()


