import json
from XMLOperation import XMLOperation
import os
import cv2

save_path = "/mnt/disk/faces/"


for i in xrange(20, 33):
    img_dir = os.path.join(save_path, str(i))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)


def transform_with_truth(truth_path, face_dir):
    position_dir = os.path.join(face_dir, "position.json")
    f = open(position_dir, "r")
    for line in f:
        position = json.loads(line)
    truth = XMLOperation.read_from_file(truth_path)

    for _, image_name in enumerate(truth):
        if image_name == "nothing":
            continue
        print("Image name: {}".format(image_name))
        datas = truth[image_name]
        for data in datas:
            id, l, t, r, b = data[0], data[2], data[3], data[4], data[5]

            faces = position[image_name]
            min_diff = 1e9
            similar_face_index = None
            for index, face in enumerate(faces):
                diff = abs(data[2] - face[0]) + abs(data[3] - face[1]) + abs(data[4] - face[2]) + abs(data[5] - face[3])
                if diff < min_diff:
                    diff = min_diff
                    similar_face_index = index
            if min_diff < 100:
                pic = cv2.imread(os.path.join(face_dir, image_name, str(similar_face_index) + ".png"))
                cv2.imwrite(os.path.join(save_path, str(id), image_name), pic)


transform_with_truth("/share/dataset/train/1_1_04_0/face recognize label result/dongnanmeneast_15_1920x1080_30.xml", "/mnt/disk/faces_east")
transform_with_truth("/share/dataset/train/1_1_04_0/face recognize label result/dongnanmenwest_16_1920x1080_30.xml", "/mnt/disk/faces_west")