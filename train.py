from utils_key import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())   # 抓取到GPU相关信息则可以使用GPU

#定义卷积和最大池化，呈现更简洁
def conv2d(inputs, filters, kernel_size, padding='same', activation=tf.nn.relu, name=''):
    if name:
        return tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=1, padding=padding,
                                activation=activation, name=name, kernel_initializer=tf.contrib.layers.xavier_initializer())
    else:
        return tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=1, padding=padding,
                                activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())

def maxpool2d(inputs):
    return tf.layers.max_pooling2d(inputs, pool_size=2, strides=2, padding='valid')

def transform(X_batch, Y_batch):
    X_data = []
    Y_data = []

    offset = 20
    for i in range(X_batch.shape[0]):
        img = X_batch[i]
        # random rotation
        degree = int(np.random.random() * offset - offset / 2)
        rad = degree / 180 * np.pi
        mat = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), degree, 1)
        img_ = cv2.warpAffine(img, mat, (img_size, img_size), borderValue=(255, 255, 255))
        # random translation
        x0 = int(np.random.random() * offset - offset / 2)
        y0 = int(np.random.random() * offset - offset / 2)
        mat = np.float32([[1, 0, x0], [0, 1, y0]])
        img_ = cv2.warpAffine(img_, mat, (img_size, img_size), borderValue=(255, 255, 255))
        # random flip
        if np.random.random() > 0.5:
            img_ = np.fliplr(img_)
            flip = True
        else:
            flip = False

        X_data.append(img_)

        points = []
        for j in range(y_dim):
            x = Y_batch[i, j, 0] * img_size
            y = Y_batch[i, j, 1] * img_size
            # random rotation
            dx = x - img_size / 2
            dy = y - img_size / 2
            x = int(dx * np.cos(rad) + dy * np.sin(rad) + img_size / 2)
            y = int(-dx * np.sin(rad) + dy * np.cos(rad) + img_size / 2)
            # random translation
            x += x0
            y += y0

            x = x / img_size
            y = y / img_size
            points.append([x, y])
        # random flip
        if flip:
            data = {features[j]: points[j] for j in range(y_dim)}
            points = []
            for j in range(y_dim):
                col = features[j]
                if col.find('left') >= 0:
                    col = col.replace('left', 'right')
                elif col.find('right') >= 0:
                    col = col.replace('right', 'left')
                [x, y] = data[col]
                x = 1 - x
                points.append([x, y])

        Y_data.append(points)

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # preprocess
    X_data = (X_data / 255. - 0.5) * 2
    Y_heatmap = []
    for i in range(Y_data.shape[0]):
        heatmaps = []
        invert_heatmap = np.ones((heatmap_size, heatmap_size))
        for j in range(Y_data.shape[1]):
            x0 = int(Y_data[i, j, 0] * heatmap_size)
            y0 = int(Y_data[i, j, 1] * heatmap_size)
            x = np.arange(0, heatmap_size, 1, float)
            y = x[:, np.newaxis]
            #cur_heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * 1.0 ** 2))
            cur_heatmap = np.exp(-((x - x0) * (x - x0) + (y - y0) * (y - y0)) / (2.0 * 1.0 * 1.0))
            heatmaps.append(cur_heatmap)
            invert_heatmap -= cur_heatmap
        heatmaps.append(invert_heatmap)
        Y_heatmap.append(heatmaps)
    Y_heatmap = np.array(Y_heatmap)
    Y_heatmap = np.transpose(Y_heatmap, (0, 2, 3, 1))  # batch_size, heatmap_size, heatmap_size, y_dim + 1
    return X_data, Y_data, Y_heatmap


#读取数据
train = pd.read_csv(os.path.join('data', 'train', 'train_changed.csv'))
#train = train[train.image_category == 'dress']#以dress为例，测试代码运行效果
#train = train[train.image_category == 'blouse']
#train = train[train.image_category == 'outwear']
#train = train[train.image_category == 'trousers']
train = train[train.image_category == 'skirt']
train = train.to_dict('records')
#dress
'''features = [
    'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
    'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right',
    'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'hemline_left', 'hemline_right']#15'''
#blouse
'''features = [
    'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
    'armpit_left', 'armpit_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
    'cuff_right_out','top_hem_left','top_hem_right']#13'''
#outwear
'''features = [
    'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right','armpit_left',
    'armpit_right', 'waistline_left', 'waistline_right','cuff_left_in','cuff_left_out', 
    'cuff_right_in', 'cuff_right_out','top_hem_left','top_hem_right']#14'''
#trousers
'''features = [
    'waistband_left','waistband_right','crotch','bottom_left_in','bottom_left_out',
    'bottom_right_in','bottom_right_out']#7'''
#skirt
features = [
    'waistband_left','waistband_right','hemline_left', 'hemline_right']#4
#all
'''features = [
    'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
    'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right','cuff_left_in',
    'cuff_left_out', 'cuff_right_in', 'cuff_right_out','top_hem_left','top_hem_right',
    'waistband_left','waistband_right','hemline_left', 'hemline_right','crotch',
    'bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out']'''
img_size = 256

#随机选一些训练数据并绘图查看关键点效果
'''r = 6
c = 6
puzzle = np.ones((img_size * r, img_size * c, 3))
random_indexs = np.random.choice(len(train), 36)
for i in range(36):
    record = train[random_indexs[i]]
    img = imread(record['image_id'])
    img = cv2.resize(img, (img_size, img_size))
    for col in features:
        if record[col + '_s'] >= 0:
            cv2.circle(img, (int(img_size * record[col + '_x']), int(img_size * record[col + '_y'])), 3, (120, 240, 120), 2)
    img = img / 255.
    r = i // 6
    c = i % 6
    puzzle[r * img_size: (r + 1) * img_size, c * img_size: (c + 1) * img_size, :] = img
plt.figure(figsize=(15, 15))
plt.imshow(puzzle)
plt.show()'''

#整理数据并分割训练集和验证集
X_train = []
Y_train = []
for i in tqdm(range(len(train))):
    record = train[i]
    img = imread(record['image_id'])
    img = cv2.resize(img, (img_size, img_size))

    y = []
    for col in features:
        y.append([record[col + '_x'], record[col + '_y']])

    X_train.append(img)
    Y_train.append(y)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

#划分训练集和验证集，其中验证集占10%
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1)

#定义CPM模型
batch_size = 16
heatmap_size = 32
stages = 6
y_dim = Y_train.shape[1]
#定义两个占位符
X = tf.placeholder(tf.float32, [None, img_size, img_size, 3], name='X')
Y = tf.placeholder(tf.float32, [None, heatmap_size, heatmap_size, y_dim + 1], name='Y')

#各阶段卷积和池化，逐渐提取特征，形成热图
stage_heatmaps = []

h0 = maxpool2d(conv2d(conv2d(X, 64, 3), 64, 3))
h0 = maxpool2d(conv2d(conv2d(h0, 128, 3), 128, 3))
h0 = maxpool2d(conv2d(conv2d(conv2d(conv2d(h0, 256, 3), 256, 3), 256, 3), 256, 3))
for i in range(6):
    h0 = conv2d(h0, 512, 3)
sub_stage = conv2d(h0, 128, 3) # batch_size, 32, 32, 128

#阶段1
h0 = conv2d(sub_stage, 512, 1, padding='valid')
h0 = conv2d(h0, y_dim + 1, 1, padding='valid', activation=None, name='stage_1')
stage_heatmaps.append(h0)

#其他阶段
for stage in range(2, stages + 1):
    h0 = tf.concat([stage_heatmaps[-1], sub_stage], axis=3)
    for i in range(5):
        h0 = conv2d(h0, 128, 7)
    h0 = conv2d(h0, 128, 1, padding='valid')
    h0 = conv2d(h0, y_dim + 1, 1, padding='valid', activation=None, name='stage_%d' % stage)
    stage_heatmaps.append(h0)

#定义损失函数和优化器
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=1000, decay_rate=0.9)#动态缩减学习率

losses = [0 for _ in range(stages)]
total_loss = 0
for stage in range(stages):
    losses[stage] = tf.losses.mean_squared_error(Y, stage_heatmaps[stage])#使用均方误差
    total_loss += losses[stage]
#增加L2正则化项
total_loss_with_reg = total_loss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-10), tf.trainable_variables())
total_loss = total_loss / stages
total_loss_with_reg = total_loss_with_reg / stages
#使用Adam优化器减小损失
optimizer = tf.contrib.layers.optimize_loss(total_loss_with_reg, global_step=global_step, learning_rate=learning_rate,
                                            optimizer='Adam', increment_global_step=True)
#训练模型
sess = tf.Session()
sess.run(tf.global_variables_initializer())

OUTPUT_DIR ='skirt'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

for stage in range(stages):
    tf.summary.scalar('loss/loss_stage_%d' % (stage + 1), losses[stage])
tf.summary.scalar('loss/total_loss', total_loss)
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(OUTPUT_DIR)

loss_valid_min = np.inf
saver = tf.train.Saver()

epochs = 100
patience = 10
for e in range(epochs):
    loss_train = []
    loss_valid = []

    X_train, Y_train = shuffle(X_train, Y_train)
    for i in tqdm(range(X_train.shape[0] // batch_size)):
        X_batch = X_train[i * batch_size: (i + 1) * batch_size, :, :, :]
        Y_batch = Y_train[i * batch_size: (i + 1) * batch_size, :, :]
        X_data, Y_data, Y_heatmap = transform(X_batch, Y_batch)
        _, ls, lr, stage_heatmaps_ = sess.run([optimizer, total_loss, learning_rate, stage_heatmaps],
                                              feed_dict={X: X_data, Y: Y_heatmap})
        loss_train.append(ls)

        if i > 0 and i % 100 == 0:
            writer.add_summary(sess.run(summary, feed_dict={X: X_data, Y: Y_heatmap}),
                               e * X_train.shape[0] // batch_size + i)
            writer.flush()
    loss_train = np.mean(loss_train)

    demo_img = (X_data[0] + 1) / 2
    demo_heatmaps = []
    for stage in range(stages):
        demo_heatmap = stage_heatmaps_[stage][0, :, :, :y_dim].reshape((heatmap_size, heatmap_size, y_dim))
        demo_heatmap = cv2.resize(demo_heatmap, (img_size, img_size))
        demo_heatmap = np.amax(demo_heatmap, axis=2)
        demo_heatmap = np.reshape(demo_heatmap, (img_size, img_size, 1))
        demo_heatmap = np.repeat(demo_heatmap, 3, axis=2)
        demo_heatmaps.append(demo_heatmap)

    demo_gt_heatmap = Y_heatmap[0, :, :, :y_dim].reshape((heatmap_size, heatmap_size, y_dim))
    demo_gt_heatmap = cv2.resize(demo_gt_heatmap, (img_size, img_size))
    demo_gt_heatmap = np.amax(demo_gt_heatmap, axis=2)
    demo_gt_heatmap = np.reshape(demo_gt_heatmap, (img_size, img_size, 1))
    demo_gt_heatmap = np.repeat(demo_gt_heatmap, 3, axis=2)

    upper_img = np.concatenate((demo_heatmaps[0], demo_heatmaps[1], demo_heatmaps[2]), axis=1)
    blend_img = 0.5 * demo_img + 0.5 * demo_gt_heatmap
    lower_img = np.concatenate((demo_heatmaps[-1], demo_gt_heatmap, blend_img), axis=1)
    demo_img = np.concatenate((upper_img, lower_img), axis=0)
    imsave(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % e), demo_img)

    X_valid, Y_valid = shuffle(X_valid, Y_valid)
    for i in range(X_valid.shape[0] // batch_size):
        X_batch = X_valid[i * batch_size: (i + 1) * batch_size, :, :, :]
        Y_batch = Y_valid[i * batch_size: (i + 1) * batch_size, :, :]
        X_data, Y_data, Y_heatmap = transform(X_batch, Y_batch)
        ls = sess.run(total_loss, feed_dict={X: X_data, Y: Y_heatmap})
        loss_valid.append(ls)
    loss_valid = np.mean(loss_valid)

    print('Epoch %d, lr %.6f, train loss %.6f, valid loss %.6f' % (e, lr, loss_train, loss_valid))
    if loss_valid < loss_valid_min:
        print('Saving model...')
        saver.save(sess, os.path.join(OUTPUT_DIR, 'cpm'))
        loss_valid_min = loss_valid
        patience = 10
    else:
        patience -= 1
        if patience == 0:
            break
