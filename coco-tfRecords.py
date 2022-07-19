# This code is inspired by https://keras.io/examples/keras_recipes/creating_tfrecords/, Credit goes to Keras documentation

import os
from os.path import exists
import json
import pprint
import tensorflow as tf
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

class cocoTfrecord:
    def __init__(self, annotation="./annotation.json", images='./images', outputDir="./outputs"):
        # Opening JSON file
        if not exists(annotation):
            print("annotation file cant be found, please check then try again")
            return
            
        # makes the output dirctory if not exist
        if not exists(outputDir):
            os.mkdir(outputDir)
        self.outputDir = outputDir
        self.image_per_file = 4096
        self.num_tfrecords = int
        with open(annotation) as json_file:
            self.data = json.loads(json_file.read(), object_hook=lambda d: SimpleNamespace(**d))
            print(str(len(self.data.categories)), "class/es")
            print(str(len(self.data.images)), "image/s")
            self.num_tfrecords = len(self.data.annotations) // self.image_per_file
            if len(self.data.annotations) % self.image_per_file:
                self.num_tfrecords += 1 
        self.images = images

    def convert(self):
        for tfrec_num in np.arange(self.num_tfrecords):
            samples = self.data.annotations[(tfrec_num * self.image_per_file).astype(int) : ((tfrec_num + 1) * self.image_per_file).astype(int)]

            with tf.io.TFRecordWriter(
                    self.outputDir + "/file_%.2i-%i.tfrecord" % (tfrec_num, len(samples))
                ) as writer:
                for sample in samples:

                    image_path = "./" + self.data.images[sample.image_id].file_name 
                    filename, file_extension = os.path.splitext(image_path)
                    if file_extension == ".png":
                        image =  tf.io.decode_png(tf.io.read_file(image_path))
                    elif file_extension == ".jpg" or  file_extension == ".jpeg":
                        image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                    example = self.create_example(image, image_path, sample)
                    writer.write(example.SerializeToString())

    def image_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
        )


    def bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


    def float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def float_feature_list(self, value):
        """Returns a list of float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    def create_example(self, image, path, example):
        feature = {
            "image": self.image_feature(image),
            "path": self.bytes_feature(path),
            "area": self.float_feature(example.area),
            "bbox": self.float_feature_list(example.bbox),
            "category_id": self.int64_feature(example.category_id),
            "id": self.int64_feature(example.id),
            "image_id": self.int64_feature(example.image_id),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example
    
if __name__ == '__main__':
    c = cocoTfrecord()
    c.convert()

    raw_dataset = tf.data.TFRecordDataset("./outputs/file_00-5.tfrecord")
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

    for features in parsed_dataset.take(1):
        for key in features.keys():
            if key != "image":
                print(f"{key}: {features[key]}")

        print(f"Image shape: {features['image'].shape}")
        plt.figure(figsize=(7, 7))
        plt.imshow(features["image"].numpy())
        plt.show()