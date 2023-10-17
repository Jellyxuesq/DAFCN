# 导入必要的库
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import nibabel as nib # 用于读取.nii文件
import os # 用于处理文件路径


# 定义深度注意力全卷积网络的函数，参考论文中的图2和表II
def deep_attention_fcn():
  # 返回一个keras模型对象，表示深度注意力全卷积网络
  # 输入是一个四维张量，形状为(batch_size, height, width, channels)，表示MRI图像，channels为1
  # 输出是一个四维张量，形状为(batch_size, height, width, classes)，表示前列腺分割的结果，classes为2（背景和前列腺）

  # 定义输入层
  inputs = layers.Input(shape=(None, None, 1))

  # 定义编码器部分，使用四个卷积块，每个卷积块包含两个卷积层和一个最大池化层，参考论文中的表II
  conv1_1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
  conv1_2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv1_1)
  pool1 = layers.MaxPooling2D((2, 2))(conv1_2)

  conv2_1 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
  conv2_2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv2_1)
  pool2 = layers.MaxPooling2D((2, 2))(conv2_2)

  conv3_1 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
  conv3_2 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv3_1)
  pool3 = layers.MaxPooling2D((2, 2))(conv3_2)

  conv4_1 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
  conv4_2 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(conv4_1)
  pool4 = layers.MaxPooling2D((2, 2))(conv4_2)

  # 定义解码器部分，使用四个上采样块，每个上采样块包含一个上采样层，一个卷积层，一个注意力机制模块和一个拼接层，参考论文中的表II
  up5 = layers.UpSampling2D((2, 2))(pool4)
  conv5_1 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(up5)

  # 定义注意力机制模块，参考论文中的公式（4）和（5）
  # 输入是两个四维张量，形状为(batch_size, height, width, channels)，表示上采样层的输出和对应的编码器层的输出
  # 输出是一个四维张量，形状与输入相同，表示注意力加权后的编码器层的输出
  def attention(x):
    # x是一个列表，包含两个四维张量
    x1 = x[0]  # 上采样层的输出
    x2 = x[1]  # 编码器层的输出
    # 计算注意力系数，参考论文中的公式（4）
    g1 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x1)
    g2 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x2)
    g = tf.nn.relu(g1 + g2)
    alpha = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(g)
    # 计算注意力加权后的编码器层的输出，参考论文中的公式（5）
    x2_att = x2 * alpha
    return x2_att

  att5 = layers.Lambda(attention)([conv5_1, conv3_2])
  merge5 = layers.Concatenate()([conv5_1, att5])

  up6 = layers.UpSampling2D((2, 2))(merge5)
  conv6_1 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(up6)
  att6 = layers.Lambda(attention)([conv6_1, conv2_2])
  merge6 = layers.Concatenate()([conv6_1, att6])

  up7 = layers.UpSampling2D((2, 2))(merge6)
  conv7_1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(up7)
  att7 = layers.Lambda(attention)([conv7_1, conv1_2])
  merge7 = layers.Concatenate()([conv7_1, att7])

  up8 = layers.UpSampling2D((2, 2))(merge7)
  conv8_1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(up8)

  # 定义输出层，使用一个卷积层，参考论文中的表II
  outputs = layers.Conv2D(2, (1, 1), padding='same', activation='softmax')(conv8_1)

  # 创建模型对象
  model = models.Model(inputs=inputs, outputs=outputs)

  # 返回模型对象
  return model


# 定义一个函数，用于从文件夹中读取所有的.nii文件，并转化为numpy数组
def load_nii_files(folder):
    # 输入是一个字符串，表示文件夹的路径
    # 输出是一个四维数组，形状为(batch_size, height, width, channels)，表示所有的.nii文件的数据

    # 获取文件夹中所有的.nii文件的路径
    file_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.nii')]

    # 创建一个空列表，用于存储每个.nii文件的数据
    data_list = []

    # 遍历每个.nii文件的路径
    for file_path in file_paths:
        # 使用nibabel库来读取.nii文件
        im = nib.load(file_path)
        # 使用get_fdata方法来获取图像的numpy数组
        np_pixel_array = im.get_fdata()
        # 将图像的numpy数组添加到列表中
        data_list.append(np_pixel_array)

    # 将列表转化为numpy数组，并增加一个维度作为channels
    data_array = np.expand_dims(np.array(data_list), axis=-1)

    # 返回numpy数组
    return data_array


# 加载训练数据和标签，使用load_nii_files函数从两个文件夹中读取所有的.nii文件，并转化为numpy数组
x_train_folder = 'x_train' # 假设x_train文件夹存放了所有病人的初始图像
y_train_folder = 'y_train' # 假设y_train文件夹存放了所有病人人工分割后的图像
x_train = load_nii_files(x_train_folder) # 形状为(batch_size, height, width, channels)，表示MRI图像，channels为1
y_train = load_nii_files(y_train_folder) # 形状为(batch_size, height, width, channels)，表示前列腺分割的标签，channels为1

# 将前列腺分割的标签转换为one-hot编码，作为模型的输出
y_train = to_categorical(y_train, num_classes=2)

# 创建深度注意力全卷积网络的模型对象
model = deep_attention_fcn()

# 编译模型，定义损失函数，优化器和评估指标，参考论文中的第三节
model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.Adam(learning_rate=0.0001), metrics=[metrics.DiceCoefficient()])

# 定义模型的回调函数，包括保存最佳模型和提前停止训练
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_dice_coefficient', mode='max', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_dice_coefficient', mode='max', patience=10)

# 训练模型，定义训练轮数，批次大小和验证集比例，参考论文中的第三节
model.fit(x_train, y_train, epochs=100, batch_size=4, validation_split=0.2, callbacks=[model_checkpoint, early_stopping])
