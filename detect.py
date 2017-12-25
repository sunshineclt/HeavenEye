import cv2
import numpy as np
import tensorflow as tf

import FaceDetection


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


# face detection parameters
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor

# restore mtcnn model
print('Creating networks and loading parameters')
gpu_memory_fraction = 1.0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = FaceDetection.create_mtcnn(sess, './model_check_point/')


# ema = tf.train.ExponentialMovingAverage(1.0)
# saver = tf.train.Saver(ema.variables_to_restore())
# model_checkpoint_path = './model_check_point/model-20160506.ckpt-500000'
# saver.restore(sess, model_checkpoint_path)
ckpt = tf.train.get_checkpoint_state('./model_check_point/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
saver.restore(sess, ckpt.model_checkpoint_path)

pic = cv2.imread("test.jpg")

find_results = []
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

if gray.ndim == 2:
    img = to_rgb(gray)

bounding_boxes, _ = FaceDetection.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

number_of_faces = bounding_boxes.shape[0]  # number of faces

for face_position in bounding_boxes:
    face_position = face_position.astype(int)

    cv2.rectangle(pic,
                  (face_position[0],
                   face_position[1]),
                  (face_position[2],
                   face_position[3]),
                  (0, 255, 0), 2)

    crop = img[face_position[1]:face_position[3], face_position[0]:face_position[2], ]
    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
    data = crop.reshape(-1, 96, 96, 3)


# # Display the resulting frame
# cv2.imshow('face_detection', pic)
# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# cv2.destroyAllWindows()
