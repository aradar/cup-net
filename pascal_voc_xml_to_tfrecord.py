#! /usr/bin/env python3
from bs4 import BeautifulSoup
from typing import Any, List

import os
import tensorflow as tf
from numpy.core.multiarray import ndarray

from object_detection.utils import dataset_util
from os import path
import cv2

flags = tf.app.flags
flags.DEFINE_string("image_path", "./images", "Path to input images")
flags.DEFINE_string("annotations_path", "./annotations", "Path to input annotation XMLs")
flags.DEFINE_string("output_path", "./", "Path to output TFRecord")
flags.DEFINE_string("output_file", "car.tfrecord", "Filename for the TFRecord")
FLAGS = flags.FLAGS


class BoundingBox:

    def __init__(self, name: str, left: float, top: float, right: float, bottom: float,
                 normalize_values: bool = False, normalize_x: int = 0, normalize_y: int = 0) -> None:
        self.name = name
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

        if normalize_values:
            self.left = left / normalize_x
            self.right = right / normalize_x
            self.top = top / normalize_y
            self.bottom = bottom / normalize_y


class Annotation:

    def __init__(self, width: int, height: int, depth: int, bounding_boxes: List[BoundingBox]) -> None:
        self.width = width
        self.height = height
        self.depth = depth
        self.bounding_boxes = bounding_boxes


def class_name_to_int(class_name: str) -> int:
    if class_name.upper() == "YELLOW_CUP":
        return 1
    elif class_name == "BLUE_CUP":
        return 2
    else:
        raise ValueError("class_name can't be {}!".format(class_name))


def create_tf_record(image_name: str, encode_image: ndarray, annotation: Annotation) -> Any:
    height = annotation.height  # Image height
    width = annotation.width  # Image width
    filename = image_name.encode()  # Filename of the image. Empty if image is not from file
    encoded_image_data = encode_image.tobytes()  # Encoded image bytes
    image_format = b"jpeg"  # b'jpeg' or b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for bounding_box in annotation.bounding_boxes:
        xmins.append(bounding_box.left)
        xmaxs.append(bounding_box.right)

        ymins.append(bounding_box.top)
        ymaxs.append(bounding_box.bottom)

        classes_text.append(bounding_box.name.encode())
        classes.append(class_name_to_int(bounding_box.name))

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_record


def read_annotation_xml_file(img_file: str) -> Annotation:
    file, _ = path.splitext(img_file)
    file += ".xml"
    file = os.path.join(FLAGS.annotations_path, file)

    if path.isfile(file):
        file_handle = open(file, "r")
        handler = file_handle.read()
        soup = BeautifulSoup(handler, "xml")

        width = int(soup.find("width").get_text())
        height = int(soup.find("height").get_text())
        depth = int(soup.find("depth").get_text())

        boxes = []
        for annotation_object in soup.find_all("object"):
            annotation_object.find("name").get_text()
            boxes.append(
                BoundingBox(
                    annotation_object.find("name").get_text(),
                    int(annotation_object.find("xmin").get_text()),
                    int(annotation_object.find("xmax").get_text()),
                    int(annotation_object.find("ymin").get_text()),
                    int(annotation_object.find("ymax").get_text()),
                    True, width, height))
    else: # default annotation
        width = 256
        height = 256
        depth = 3
        boxes = []

    return Annotation(width, height, depth, boxes)


def main(_):
    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, FLAGS.output_file))

    image_path = path.normpath(FLAGS.image_path) + "/"
    for root, dirs, filenames in os.walk(image_path):
        for file in filenames:
            full_filename = path.join(root, file).replace(image_path, "", 1)
            annotation = read_annotation_xml_file(full_filename)
            encoded_image: ndarray = cv2.imread(path.join(root, file))
            writer.write(create_tf_record(file, encoded_image, annotation).SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
