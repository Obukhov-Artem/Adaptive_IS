from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas, numpy
from keras.models import Sequential, Model
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scale_features_mm = MinMaxScaler()

def minmax(x):
    if x<0:
        return 0
    if x>1:
        return 1
    return x

def nullmin(x):
    if x<1:
        return 1
    return x

def insert_all():

    import random

    num = 0
    n_oper = 9
    n_user = 100
    n_ex = 5
    data_x = []
    data_y = []
    operation = []
    users = []
    for j in range(n_oper):
        o_diff = (j+1)*0.1
        o_act = o_diff*10+random.randint(-3,6)
        operation.append([o_diff, o_act])
    for i in range(n_user):
        user_talent = random.random()
        user_age = random.randint(20,60)
        users.append([user_talent,user_age])
    for o in range(n_oper):
        for u in range(n_user):
            o_diff, o_act = operation[o]
            user_talent, user_age = users[u]

            for c in range(n_ex):
                random_factor = (random.random() - 0.5) * 0.1
                f = minmax((1 - o_diff) + 0.5*user_talent+(o_act/user_age)+random_factor)
                t = 1/f+10*nullmin(o_act*o_diff - user_talent - (o_act/user_age)+random_factor)
                num += 1
                data_x.append([u,user_talent, user_age, o,o_diff, o_act])
                data_y.append([100*f, t])


    print(num)

    from sklearn.model_selection import train_test_split as t_t_split
    x, y = np.array(data_x), np.array(data_y)
    X_train, X_test, y_train, y_test = t_t_split(x, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = insert_all()
for i in range(30):
    print(X_train[i], y_train[i])
epochs = 30
input = Input(shape=(6,))
h1 = Dense(300, activation="relu")(input)
h1 = Dropout(0.5)(h1)
h1 = Dense(200, activation="relu")(h1)
h1 = Dropout(0.3)(h1)
h1 = Dense(200, activation="relu")(h1)
h1 = Dense(100, activation="relu")(h1)
out_1 = Dense(2, activation='relu', name="f_layer")(h1)
model = Model(input, out_1)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                    batch_size=10,
                    epochs=epochs,
                    validation_data=(X_test, y_test))
plt.figure(figsize=(12, 6))
model.save("dd_model.h5")
r = plt.subplot(1, 2, 1)
plt.grid(True)
plt.plot(np.arange(1, epochs + 1), history.history["loss"], label="loss")
# plt.plot(np.arange(1, N+1), history.history["accuracy"], label="accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
r.set_title("Loss", fontsize=12)




r = plt.subplot(1, 2, 2)
N = epochs
plt.grid(True)
plt.plot(np.arange(1, N + 1), history.history['accuracy'], label="accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

r.set_title("Accuracy", fontsize=12)
r.set_ylim([0,1])
plt.savefig('compare.png', dpi=300)
plt.show()