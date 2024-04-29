from utils_key import *
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())   # 抓取到GPU相关信息则可以使用GPU



test = pd.read_csv(os.path.join('data', 'test', 'test.csv'))
test['image_id'] = test['image_id'].apply(lambda x: os.path.join('data','test', x))
test = test[test.image_category == 'dress']
#test = test[test.image_category == 'blouse']
#test = test[test.image_category == 'outwear']
#test = test[test.image_category == 'trousers']
#test = test[test.image_category == 'skirt']
test = test['image_id'].values

batch_size = 16
img_size = 256
test = shuffle(test)
test = test[:batch_size]
X_test = []
for i in range(batch_size):
    img = imread(test[i])
    img = cv2.resize(img, (img_size, img_size))
    X_test.append(img)
X_test = np.array(X_test)
print(X_test.shape)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

OUTPUT_DIR = 'dress'
saver = tf.train.import_meta_graph(os.path.join(OUTPUT_DIR, 'cpm.meta'))
saver.restore(sess, tf.train.latest_checkpoint(OUTPUT_DIR))

stages = 6
y_dim = 15#根据训练的数据进行调整
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
for i in range(5):
    print(i)
    print(coords[i])

n = int(np.sqrt(batch_size))
puzzle = np.ones((img_size * n, img_size * n, 3))
for i in range(batch_size):
    img = X_test[i]
    r = i // n
    c = i % n
    puzzle[r * img_size: (r + 1) * img_size, c * img_size: (c + 1) * img_size, :] = img
plt.figure(figsize=(12, 12))
plt.imshow(puzzle)
plt.show()
imsave(os.path.join(OUTPUT_DIR,'test_result.jpg'), puzzle)
