#!/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import cv2
from os import listdir
from bs4 import BeautifulSoup
from os import path
from tqdm import tqdm

# Set Logging
tf.logging.set_verbosity(tf.logging.INFO)

# set constants
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
NUMBER_OF_COLORS = 3

PREDICTION_DIVIDER = 4

PREDICTION_IMAGE_WIDTH = int(INPUT_IMAGE_WIDTH / PREDICTION_DIVIDER)
PREDICTION_IMAGE_HEIGHT = int(INPUT_IMAGE_HEIGHT / PREDICTION_DIVIDER)

BATCH_SIZE = 1

IMAGE_DIR = "images/"
XML_DIR = "xmls/"
IMAGE_END = ".jpg"

MODE = "TRAIN"
#MODE = "TEST"
#MODE = "EVAL"

LEARNING_RATE = 0.000002
STEPS_TO_LEARN = 100
LABEL_VALUE = 1.0
DATA_SCALE = 1.0

NUMBER_FILTERS1 = 10
NUMBER_FILTERS2 = 10

def cnn_model_fn(features, labels, mode):
    # input layer
    input_layer = tf.reshape(features["x"], [-1, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, NUMBER_OF_COLORS])

    #input_layer = tf.Print(input_layer, [input_layer], "inputs:", summarize=1000)

    #print("-------------------------------------")
    #print("features[x]:", features["x"])
    #print("-------------------------------------")

    # convolutional layer 2
    conv1 = tf.layers.conv2d(inputs = input_layer, filters=NUMBER_FILTERS1, kernel_size=[10, 10], padding="same", activation=tf.nn.relu)

    # pool1
    # pool1.shape == (BATCHSIZE(20), 128, 128, NUMBER_OF_FILTERS(32))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #pool1 = tf.Print(pool1, [pool1], "pool1: ")

    # convolutional layer 2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=NUMBER_FILTERS2, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # pool2
    # pool2.shape == (BATCHSIZE(20), 64, 64, NUMBER_OF_FILTERS(16))
    POOL2_COMPRESSION = 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2*POOL2_COMPRESSION, 2*POOL2_COMPRESSION], strides=2*POOL2_COMPRESSION)
    #pool2 = tf.Print(pool2, [pool2], "pool2: ")

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, int(INPUT_IMAGE_WIDTH/(4*POOL2_COMPRESSION)) * int(INPUT_IMAGE_HEIGHT/(4*POOL2_COMPRESSION)) * NUMBER_FILTERS2])
    #dense = tf.layers.dense(inputs=pool2_flat, units=64*64*16, activation=tf.nn.relu)
    dense = tf.layers.dense(inputs=pool2_flat, units=15000, activation=tf.nn.relu)
    #dropout = tf.layers.dropout(inputs=dense, rate=0.1, training=(mode == tf.estimator.ModeKeys.TRAIN))

    #dropout = tf.Print(dropout, [dropout], message="dropout: ")

    # Logits Layer
    # we have INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT * 2 ouput neurons
    # logits.shape == (10, INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT * 2(Number_of_objects)
    logits = tf.layers.dense(inputs=dense, units=PREDICTION_IMAGE_WIDTH * PREDICTION_IMAGE_HEIGHT * 2, activation=tf.nn.sigmoid)

    logits = tf.Print(logits, [tf.train.get_global_step(), logits], message="logits: ")

    predictions = {
        # Generate Predictions Pixels
        "pixels": tf.reshape(logits, [-1, PREDICTION_IMAGE_WIDTH*2, PREDICTION_IMAGE_HEIGHT, 1], name="predictions"),
        # Add 'soft_max_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'.
        # "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        #print("---------------------------")
        #print("PREDICT")
        #print("---------------------------")
        return tf.estimator.EstimatorSpec(logits, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy = -tf.reduce_sum(((labels*tf.log(logits + 1e-9)) + ((1-labels) * tf.log(1 - logits + 1e-9))), name='xentropy') / (PREDICTION_IMAGE_WIDTH * 2 * PREDICTION_IMAGE_HEIGHT)
    loss = cross_entropy
    #loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.cast(labels, tf.float32), logits=logits)

    #print("---------------------------")
    #print("logits.shape:")
    #print(logits.shape)
    #print("---------------------------")

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    #print("---------------------------")
    #print("EVALUATE")
    #print("---------------------------")
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=tf.reshape(labels, [-1, PREDICTION_IMAGE_WIDTH*2, PREDICTION_IMAGE_HEIGHT, 1]), predictions=predictions["pixels"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Loading Data

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

    def to_string(self):
        return "left: " + str(self.left) + "  right:" + str(self.right) + "  top:" + str(self.top) + "  bottom:" + str(self.bottom)


class Annotation:
    def __init__(self, width: int, height: int, depth: int, bounding_boxes) -> None:
        self.width = width
        self.height = height
        self.depth = depth
        self.bounding_boxes = bounding_boxes


def read_annotation_xml_file(xml_path: str) -> Annotation:
    if path.isfile(xml_path):
        file_handle = open(xml_path, "r")
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
                    int(annotation_object.find("ymin").get_text()),
                    int(annotation_object.find("xmax").get_text()),
                    int(annotation_object.find("ymax").get_text()),
                    True, width, height))
    else: # default annotation
        width = PREDICTION_IMAGE_WIDTH
        height = PREDICTION_IMAGE_HEIGHT
        depth = NUMBER_OF_COLORS
        boxes = []

    return Annotation(width, height, depth, boxes)

class DataSet:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

def load_image(path):
    if NUMBER_OF_COLORS == 1:
        train_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(INPUT_IMAGE_WIDTH*INPUT_IMAGE_HEIGHT*NUMBER_OF_COLORS)
    elif NUMBER_OF_COLORS == 3:
        train_data = cv2.imread(path).reshape(INPUT_IMAGE_WIDTH*INPUT_IMAGE_HEIGHT*NUMBER_OF_COLORS)
    else:
        print("invalid number of Colors:", NUMBER_OF_COLORS)

    train_data = train_data * DATA_SCALE

    return train_data


def create_data(path, file_list):
    data = np.empty((len(file_list), INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT * NUMBER_OF_COLORS), dtype=np.float32)
    i = 0
    for image_path in tqdm(file_list):
        #print("  loading", path + IMAGE_DIR + image_path)
        image_data = np.array([load_image(path + IMAGE_DIR + image_path)], dtype=np.float32)
        data[i] = image_data
        i += 1

    return data


def create_label(bounding_boxes):
    label = np.zeros((PREDICTION_IMAGE_WIDTH, PREDICTION_IMAGE_HEIGHT), dtype=np.float32)
    for box in bounding_boxes:
        img_left = int(PREDICTION_IMAGE_WIDTH * box.left)
        img_right = int(PREDICTION_IMAGE_WIDTH * box.right)
        img_top = int(PREDICTION_IMAGE_HEIGHT * box.top)
        img_bottom = int(PREDICTION_IMAGE_WIDTH * box.bottom)
        #print("left: " + str(img_left) + "  right:" + str(img_right) + "  top:" + str(img_top) + "  bottom:" + str(img_bottom))
        for x in range(img_left, img_right):
            for y in range(img_top, img_bottom):
                label[y, x] = LABEL_VALUE

    return label.reshape(PREDICTION_IMAGE_WIDTH * PREDICTION_IMAGE_HEIGHT)

def create_labels(path, xml_file_list):
    labels = np.empty((len(xml_file_list), PREDICTION_IMAGE_WIDTH * PREDICTION_IMAGE_HEIGHT * 2), dtype=np.float32)
    i = 0
    for xml_file in tqdm(xml_file_list):
        #print("  loading", path + XML_DIR + xml_file)
        annotation = read_annotation_xml_file(path + XML_DIR + xml_file)

        yellow_boxes = [box for box in annotation.bounding_boxes if box.name == "YELLOW_CUP"]
        blue_boxes = [box for box in annotation.bounding_boxes if box.name == "BLUE_CUP"]

        yellow_label = create_label(yellow_boxes)
        blue_label = create_label(blue_boxes)


        concated_label = np.concatenate((yellow_label, blue_label))

        labels[i] = concated_label
        i += 1

    return labels


def create_data_set(path):
    image_file_list = listdir(path + IMAGE_DIR)

    # filter with IMAGE_END
    image_file_list = [x for x in image_file_list if x.endswith(IMAGE_END)]
    xml_file_list = [x[:-4] + ".xml" for x in image_file_list]

    data = create_data(path, image_file_list)
    labels = create_labels(path, xml_file_list)

    return DataSet(data, labels)


def show_images(images):
    counter = 0
    for img in images:
        counter += 1
        cv2.imshow("Test" + str(counter), img)
    while cv2.waitKey() != 27:
        pass
    cv2.destroyAllWindows()


def check_data_set(data_set):
    assert not np.any(np.isnan(data_set.data))
    assert not np.any(np.isnan(data_set.labels))


def print_data_set(data_set):
    print("data_set.data.shape:")
    print(data_set.data[0].shape)

    labels = data_set.labels[0].reshape(256*2, 256)
    show_images([data_set.data[0].reshape(256, 256, 3), labels])


def main(unused_argv):
    # Loading Data
    print("loading train data:")
    if MODE == "TRAIN":
        train_data_set = create_data_set("CUP-data/train/")
    elif MODE == "TEST":
        train_data_set = create_data_set("CUP-data/test/")
    elif MODE == "EVAL":
        pass

    if MODE is not "EVAL":
        check_data_set(train_data_set)

    print("loading evaluation data:")
    eval_data_set = create_data_set("CUP-data/eval/")

    check_data_set(eval_data_set)

    eval_data = eval_data_set.data
    #eval_labels = np.asarray([eval_data_set.labels])
    eval_labels = eval_data_set.labels.reshape((-1, 2 * PREDICTION_IMAGE_WIDTH * PREDICTION_IMAGE_HEIGHT))

    print("eval_data.shape:")
    print(eval_data.shape)

    print("eval_labels.shape:")
    print(eval_labels.shape)

    """
    # print evaluation
    for i in range(eval_labels.shape[0]):
        l = eval_labels[i]
        d = eval_data[i]

        #print("predictions.shape:", predictions.shape)
        cv2.imshow("Labels", l.reshape(PREDICTION_IMAGE_WIDTH*2, PREDICTION_IMAGE_HEIGHT))
        cv2.imshow("Image", d.reshape(INPUT_IMAGE_WIDTH, INPUT_IMAGE_WIDTH, 3))

        cv2.moveWindow("Labels", 1000, 300)
        cv2.moveWindow("Image", 600, 300)
        i += 1

        k = cv2.waitKey()
        if k == 27:
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()
    """

    # Create the Estimator
    cup_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./__model_checkpoints__")

    # Set up logging for predictions
    tensors_to_log = {"pixels": "predictions"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=20)

    # Train the model

    if MODE == "TRAIN" or MODE == "TEST":
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data_set.data}, y=train_data_set.labels, batch_size=BATCH_SIZE, num_epochs=None, shuffle=True)
    if MODE == "TRAIN":
        cup_classifier.train(input_fn=train_input_fn, steps=STEPS_TO_LEARN)
    elif MODE == "TEST":
        cup_classifier.train(input_fn=train_input_fn, steps=1)
    elif MODE == "EVAL":
        pass

    # Evaluate the model and print results
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    # eval_results = cup_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)

    # Predict
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, num_epochs=1, shuffle=False)
    pred_list = list(cup_classifier.predict(input_fn=predict_input_fn))
    i = 0
    print("len(pred_list):", len(pred_list))
    go_out = False
    for p in pred_list:
        predictions = p["pixels"]

        cv2.imshow("Predictions", np.kron(predictions.reshape(PREDICTION_IMAGE_WIDTH*2, PREDICTION_IMAGE_HEIGHT), np.ones((PREDICTION_DIVIDER, PREDICTION_DIVIDER)) ))
        cv2.imshow("Data", eval_data[i].reshape(INPUT_IMAGE_WIDTH, INPUT_IMAGE_WIDTH, 3) * (1/(256*DATA_SCALE)))

        cv2.moveWindow("Predictions", 1000, 300)
        cv2.moveWindow("Data", 600, 300)
        i += 1

        k = cv2.waitKey()
        while (k != 32) and not go_out:
            if k == 27:
                cv2.destroyAllWindows()
                go_out = True
                break
            k = cv2.waitKey()
        cv2.destroyAllWindows()

        if go_out:
            break

if __name__ == "__main__":
    tf.app.run()
