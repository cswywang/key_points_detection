from utils_key import *

#对训练图片进行数据增强，使得其经过随机旋转、随机平移、随机水平翻转后，关键点仍能对应良好，得到适应更多环境的CPM模型
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
            cur_heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * 1.0 ** 2))
            heatmaps.append(cur_heatmap)
            invert_heatmap -= cur_heatmap
        heatmaps.append(invert_heatmap)
        Y_heatmap.append(heatmaps)
    Y_heatmap = np.array(Y_heatmap)
    Y_heatmap = np.transpose(Y_heatmap, (0, 2, 3, 1))  # batch_size, heatmap_size, heatmap_size, y_dim + 1

    return X_data, Y_data, Y_heatmap

#读取数据
train = pd.read_csv(os.path.join('data', 'train', 'train_changed.csv'))
train = train[train.image_category == 'dress']#以dress为例，测试代码运行效果
train = train.to_dict('records')
features = [
    'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
    'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right','cuff_left_in',
    'cuff_left_out', 'cuff_right_in', 'cuff_right_out','top_hem_left','top_hem_right',
    'waistband_left','waistband_right','hemline_left', 'hemline_right','crotch',
    'bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out']

img_size = 256
batch_size = 16
heatmap_size = 32
stages = 6
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

y_dim = Y_train.shape[1]

#对数据增强后的训练集图片查看其关键点的对应效果
X_batch = X_train[:batch_size]
Y_batch = Y_train[:batch_size]
X_data, Y_data, Y_heatmap = transform(X_batch, Y_batch)

n = int(np.sqrt(batch_size))
puzzle = np.ones((img_size * n, img_size * n, 3))
for i in range(batch_size):
    img = (X_data[i] + 1) / 2
    for j in range(y_dim):
        cv2.circle(img, (int(img_size * Y_data[i, j, 0]), int(img_size * Y_data[i, j, 1])), 3, (120, 240, 120), 2)
    r = i // n
    c = i % n
    puzzle[r * img_size: (r + 1) * img_size, c * img_size: (c + 1) * img_size, :] = img
plt.figure(figsize=(12, 12))
plt.imshow(puzzle)
plt.show()