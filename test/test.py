import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r"K:\DataModel\CREDIT_SCORE\data\scaler_data\standar_data.csv", encoding='utf-8')

# 划分自变量与因变量
x, y = data.iloc[:, 1:-1], data.iloc[:, -1]

x = pd.DataFrame(MinMaxScaler().fit_transform(x))

# 建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(28, ), activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x, y, epochs=10)

model.save(r"K:\DataModel\CREDIT_SCORE\data\scaler_data\model.h5")