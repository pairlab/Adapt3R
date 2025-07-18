import os
import torch

from libero.libero.benchmark import Benchmark, grab_language_from_filename, Task, register_benchmark
from adapt3r.envs.libero.custom_task_map import libero_custom_task_map

libero_suites = [
    "libero_90_distractor",
    "libero_10_distractor",
    "libero_90_sawyer",
    "libero_10_sawyer",
    "libero_90_ur5e",
    "libero_10_ur5e",
    "libero_90_kinova3",
    "libero_10_kinova3",
    "libero_90_iiwa",
    "libero_10_iiwa",
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    for task in libero_custom_task_map[libero_suite]:
        language = grab_language_from_filename(task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.init",
        )

    
def get_libero_custom_path():
    return os.path.dirname(os.path.abspath(__file__))


class CustomBenchmark(Benchmark):

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        self.tasks = tasks
        self.n_tasks = len(self.tasks)

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_custom_path(),
            'bddl_files',
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_init_states_path(self, i):
        init_states_path = os.path.join(
            get_libero_custom_path(),
            'init_states',
            self.tasks[i].problem_folder,
            self.tasks[i].init_states_file,
        )
        return init_states_path

    def get_task_init_states(self, i):
        init_states_path = self.get_task_init_states_path(i)
        init_states = torch.load(init_states_path, weights_only=False)
        return init_states


@register_benchmark
class LIBERO_90_DISTRACTOR(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90_distractor"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_DISTRACTOR(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_distractor"
        self._make_benchmark()


@register_benchmark
class LIBERO_90_SAWYER(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90_sawyer"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_SAWYER(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_sawyer"
        self._make_benchmark()


@register_benchmark
class LIBERO_90_UR5E(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90_ur5e"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_UR5E(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_ur5e"
        self._make_benchmark()


@register_benchmark
class LIBERO_90_KINOVA3(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90_kinova3"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_KINOVA3(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_kinova3"
        self._make_benchmark()


@register_benchmark
class LIBERO_90_IIWA(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90_iiwa"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_IIWA(CustomBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_iiwa"
        self._make_benchmark()
