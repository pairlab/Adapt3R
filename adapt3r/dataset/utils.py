import os
import shutil
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def copy_file(src, dst):
    """Function to copy a single file from src to dst with optimized buffer size for large files."""
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # Use a larger buffer size (8MB) for more efficient copying of large files
        shutil.copy2(src, dst, follow_symlinks=False)
    except Exception as e:
        print(f"Error copying {src}: {e}")
        
def get_all_files(src_dir, verbose=False):
    if verbose:
        print(f'searching {src_dir}')
    files = []
    for f in os.listdir(src_dir):
        if os.path.isfile(os.path.join(src_dir, f)):
            files.append(f)
        else:
            inner_files = get_all_files(os.path.join(src_dir, f))
            for inner_file in inner_files:
                files.append(os.path.join(f, inner_file))
    return files

def copy_files_parallel(src_dir, dst_dir, num_threads=4, verbose=False):
    """
    Function to copy files from src_dir to dst_dir using parallelism.

    :param src_dir: Source directory where the files are located
    :param dst_dir: Destination directory where the files should be copied
    :param num_threads: Number of parallel threads to use for copying
    """
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # List all files in the source directory
    # files_to_copy = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    files_to_copy = get_all_files(src_dir)

    # Prepare the source and destination file paths
    src_dst_pairs = [(os.path.join(src_dir, f), os.path.join(dst_dir, f)) for f in files_to_copy]

    # Use ThreadPoolExecutor to copy files in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit copy tasks to the thread pool
        futures = [executor.submit(lambda pair: copy_file(pair[0], pair[1]), pair) for pair in src_dst_pairs]

        with tqdm(total=len(futures), disable=not verbose) as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                pbar.update(1)
        # executor.map(lambda pair: copy_file(pair[0], pair[1]), src_dst_pairs)
