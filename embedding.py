from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

import facenet


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read the file containing the pairs used for testing
            train_set = facenet.get_dataset(args.data_dir)

            # Get the paths for the corresponding images
            paths, labels = facenet.get_image_paths_and_labels(train_set)
            print(paths, labels, len(train_set))
            # Load the model
            facenet.load_model(args.model)
            print(paths)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            nrof_images = len(paths)
            emb_array = np.zeros((nrof_images, embedding_size))
            images = facenet.load_data(paths, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[0:nrof_images, :] = sess.run(embeddings, feed_dict=feed_dict)
            np.save("20.npy", emb_array)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
