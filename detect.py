import cv2
import numpy as np
import tensorflow as tf
import os

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

path = "/share/dataset/train/1_1_04_0/prob/dongnanmeneast_15_1920x1080_30/"
save_path = "/mnt/disk/faces/"
# path = "."
# save_path = "./test/"

files = os.listdir(path)
for file_name in files:
    print("Now detecting face in: ", file_name)
    if ".jpg" not in file_name:
        continue

    pic = cv2.imread(os.path.join(path, file_name))

    find_results = []
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    img = to_rgb(gray)

    bounding_boxes, _ = FaceDetection.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    number_of_faces = bounding_boxes.shape[0]  # number of faces

    index = 0
    img_dir = os.path.join(save_path, file_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
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
        cv2.imwrite(os.path.join(img_dir, str(index) + ".png"), crop)
        index += 1



        # Display the resulting frame
        # cv2.imshow('face_detection', pic)
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        # # When everything is done, release the capture
        # cv2.destroyAllWindows()
