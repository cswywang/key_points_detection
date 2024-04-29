import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from imageio import imread, imsave
import os
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())   # 抓取到GPU相关信息则可以使用GPU




def test(image_path,category):

    batch_size = 1 # 处理一张图片
    img_size = 256

    X_test = []
    img = imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    X_test.append(img)
    X_test = np.array(X_test)
    print(X_test.shape)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    OUTPUT_DIR = f"F:\key_points\\{category}"
    
    saver = tf.train.import_meta_graph(os.path.join(OUTPUT_DIR, 'cpm.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(OUTPUT_DIR))

    stages = 6
    if category == 'dress':
        y_dim = 15
    elif category == 'blouse':
        y_dim = 13
    elif category == 'outwear':
        y_dim = 14
    elif category == 'trousers':
        y_dim = 7
    elif category == 'skirt':
        y_dim = 4
    else:
        print("Category is wrong. Please choose category in [dress,blouse,outwear,trousers,skirt].")
        return 

    heatmap_size = 32
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    stage_heatmap = graph.get_tensor_by_name('stage_%d/BiasAdd:0' % stages)#stage_%d/BiasAdd:0换为stage_%d

    def visualize_result(imgs, heatmap, joints):
        imgs = imgs.astype(np.int32)
        coords = []
        for i in range(imgs.shape[0]):
            hp = heatmap[i, :, :, :joints].reshape((heatmap_size, heatmap_size, joints))
            hp = cv2.resize(hp, (img_size, img_size))
            coord = np.zeros((joints, 2))

            for j in range(joints):
                xy = np.unravel_index(np.argmax(hp[:, :, j]), (img_size, img_size))
                coord[j, :] = [xy[0], xy[1]]
                cv2.circle(imgs[i], (xy[1], xy[0]), 3, (120, 240, 120), 2)

            coords.append(coord)

        return imgs / 255., coords

    heatmap = sess.run(stage_heatmap, feed_dict={X: (X_test / 255. - 0.5) * 2})
    X_test, coords = visualize_result(X_test, heatmap, y_dim)
    print(coords[0])
    """plt.imshow(X_test[0])
    plt.show()"""

    n = int(np.sqrt(batch_size))
    puzzle = np.ones((img_size * n, img_size * n, 3))
    for i in range(batch_size):
        img = X_test[i]
        r = i // n
        c = i % n
        puzzle[r * img_size: (r + 1) * img_size, c * img_size: (c + 1) * img_size, :] = img
    plt.figure(figsize=(6, 6))
    plt.imshow(puzzle)
    #plt.show()
    imsave(r"F:\key_points\\data\test_result.jpg", puzzle)

    return coords[0]# 关键点坐标

#test(r"F:\key_points\data\test\Images\\blouse\\00b1a0e756de3804e560b84062e0fb8e.jpg",'blouse')#调用示例