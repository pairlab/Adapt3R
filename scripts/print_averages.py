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
    prefix = os.path.commonprefix(args.data_dirs)
    for data_dir in args.data_dirs:
        data = []
        for root, dirs, files in os.walk(data_dir):
            if 'data.json' in files:
                if args.filter is not None and args.filter not in root:
                    continue
                with open(os.path.join(root, 'data.json'), 'r') as f:
                    data_dict = json.load(f)
                data.append(data_dict['rollout']['overall_success_rate'])
        mean = np.mean(data)
        std_error = np.std(data) / np.sqrt(len(data))
        mean_success_rate.append(mean)
        std_errors.append(std_error)

        print(f'{data_dir[len(prefix):]}: {mean:1.3f} +/- {std_error:1.3f}; len: {len(data)}')


if __name__ == '__main__':
    main()