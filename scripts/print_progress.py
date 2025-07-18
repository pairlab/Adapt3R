from argparse import ArgumentParser
import os
import json
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument('data_dirs', nargs='*')
    parser.add_argument('--filter')
    args = parser.parse_args()

    mean_success_rate = []
    std_errors = []
    progresses = []
    prefix = os.path.commonprefix(args.data_dirs)
    for data_dir in args.data_dirs:
        data, progress = [], []
        for root, dirs, files in os.walk(data_dir):
            if 'progress.json' in files:
                if args.filter is not None and args.filter not in root:
                    continue
                with open(os.path.join(root, 'progress.json'), 'r') as f:
                    data_dict = json.load(f)
                successes = data_dict['successes']
                num = len(successes)
                progress.append(num)
                data.append(np.mean(successes))
        mean = np.median(data)
        std_error = np.std(data) / np.sqrt(len(data))
        avg_progress = np.mean(progress)
        mean_success_rate.append(mean)
        std_errors.append(std_error)

        print(f'{data_dir[len(prefix):]}: {mean:1.3f} +/- {std_error:1.3f} | envs_evaled: {avg_progress:3.1f}')


if __name__ == '__main__':
    main()