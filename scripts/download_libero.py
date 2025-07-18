import os

import libero.libero.utils.download_utils as download_utils


def main():

    os.makedirs('data/libero', exist_ok=True)
    print(f"Datasets downloaded to data/libero")
    print(f"Downloading libero_100 datasets")

    download_utils.libero_dataset_download(
        download_dir='data/libero', datasets='libero_100'
    )

    os.rename('data/libero/libero_10', 'data/libero/libero_10_unprocessed')
    os.rename('data/libero/libero_90', 'data/libero/libero_90_unprocessed')


if __name__ == "__main__":
    main()