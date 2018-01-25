#! /usr/bin/env python3

"""
Utility to copy the raw dataset images into specific evaluation and training folders.

SOURCE_DIR must contain an annotations and images dir!

Usage:
  build_image_structure SOURCE_DIR OUTPUT_DIR [-c NUM_FILES_USED] [-s EVAL_SPLIT] [-e EVAL_DIR_NAME] [-t TRAIN_DIR_NAME] [--strict]

Options:
  -c NUM_FILES_USED   Number of files to use [Default: 1000].
  -s EVAL_SPLIT       Percentage of evaluation images based on the NUM_FILES_USED [Default: 0.1].
  -e EVAL_DIR_NAME    Name of the dir in which evaluation pairs get copied [Default: eval].
  -t TRAIN_DIR_NAME   Name of the dir in which training pairs get copied [Default: train].
  --strict            Allows only images with a corresponding annotation [Default: False].
"""
import os
import random
import shutil
import sys
from docopt import docopt
from typing import Dict, List


def create_dir(dir_name: str) -> bool:
    created_dir: bool = False
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        created_dir = True
    return created_dir


def copy_files(file_keys: List[str], file_to_paths: Dict[str, Dict[str, str]], dest_path: str) -> None:
    for file_key in file_keys:
        for file_type in file_to_paths[file_key].keys():
            file_path = file_to_paths[file_key][file_type]
            _, filename = os.path.split(file_path)
            shutil.copyfile(file_path, os.path.join(dest_path, os.path.join(file_type, filename)))


if __name__ == '__main__':
    arguments = docopt(__doc__)

    files_to_use = int(arguments["-c"])
    eval_split = float(arguments["-s"])

    eval_pic_size = int(files_to_use * eval_split)
    train_pic_length = files_to_use - eval_pic_size

    print("using evaluation pictures: {}".format(eval_pic_size))
    print("using training pictures: {}".format(train_pic_length))

    source_dir = arguments["SOURCE_DIR"]
    print("using SOURCE_DIR: {}".format(source_dir))

    name_to_paths: Dict[str, Dict[str, str]] = {}
    for root, dirs, filenames in os.walk(os.path.join(source_dir, "images")):
        for filename in filenames:
            file_without_ext, _ = os.path.splitext(filename)
            name_to_paths[file_without_ext] = {"image": os.path.join(root, filename)}
    for root, dirs, filenames in os.walk(os.path.join(source_dir, "annotations")):
        for filename in filenames:
            file_without_ext, _ = os.path.splitext(filename)
            name_to_paths[file_without_ext]["annotation"] = os.path.join(root, filename)
    print("found {} pictures in the source dir".format(len(name_to_paths)))

    strict_mode = bool(arguments["--strict"])
    if strict_mode:
        print("using strict mode")
        marked_keys = []
        for name, paths in name_to_paths.items():
            if len(paths) != 2:
                marked_keys.append(name)
        for key in marked_keys:
            del name_to_paths[key]
    if len(name_to_paths) <= files_to_use:
        print("there are not enough images in the source dir! exiting...")
        sys.exit(-1)

    shuffled_data_keys = list(name_to_paths.keys())
    random.shuffle(shuffled_data_keys)
    shuffled_eval_keys = shuffled_data_keys[:eval_pic_size]
    shuffled_train_keys = shuffled_data_keys[eval_pic_size:train_pic_length + eval_pic_size]

    output_dir = arguments["OUTPUT_DIR"]
    print("using OUTPUT_DIR: {}".format(output_dir))

    train_dir = os.path.join(output_dir, arguments["-t"])
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    create_dir(train_dir)

    eval_dir = os.path.join(output_dir, arguments["-e"])
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    create_dir(eval_dir)

    print("copying data")
    for dir in [eval_dir, train_dir]:
        create_dir(os.path.join(dir, "annotation"))
        create_dir(os.path.join(dir, "image"))
    copy_files(shuffled_eval_keys, name_to_paths, eval_dir)
    copy_files(shuffled_train_keys, name_to_paths, train_dir)
