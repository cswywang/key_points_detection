from utils_key import *

train = pd.read_csv(os.path.join('data', 'train', 'train.csv'))
#print(len(train))
#print(train.head())
train['image_id'] = train['image_id'].apply(lambda x:os.path.join('data','train', x))#更换训练图片目录
#print(train.head())
#train.to_csv(os.path.join('data', 'train','train_changed.csv'), index=False)#保存

#修改关键点的属性值，将其由一个属性的三元组变成3个属性的一元值
columns = train.columns
for col in columns:
    if col in ['image_id', 'image_category']:
        continue
    train[col + '_x'] = train[col].apply(lambda x:float(x.split('_')[0]))
    train[col + '_y'] = train[col].apply(lambda x:float(x.split('_')[1]))
    train[col + '_s'] = train[col].apply(lambda x:float(x.split('_')[2]))
    train.drop([col], axis=1, inplace=True)
#print(train.head())

#对所有服饰的坐标进行归一化，得出关键点在图像中的相对位置
features = [
    'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
    'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right','cuff_left_in',
    'cuff_left_out', 'cuff_right_in', 'cuff_right_out','top_hem_left','top_hem_right',
    'waistband_left','waistband_right','hemline_left', 'hemline_right','crotch',
    'bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out']

train = train.to_dict('records')
for i in tqdm(range(len(train))):
    record = train[i]
    img = imread(record['image_id'])
    h = img.shape[0]
    w = img.shape[1]
    for col in features:
        if record[col + '_s'] >= 0:
            train[i][col + '_x'] /= w
            train[i][col + '_y'] /= h
        else:
            train[i][col + '_x'] = 0
            train[i][col + '_y'] = 0

train_df = pd.DataFrame(train)
print(train_df.head())
train_df.to_csv(os.path.join('data', 'train','train_changed.csv'), index=False)#保存
