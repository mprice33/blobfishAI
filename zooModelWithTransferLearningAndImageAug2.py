# Bringing the combination of transfer learning and image augmentation together to create high quality model
# Using InceptionResnetV2, led to best results.
# All previous alterations can be tried here, proper epoch amount can be found through testing
# These were the final values in which we saw the best results, many intermediate values were not recorded
# No layers were unfrozen but may find better results by freezing/unfreezing specific layers for retraining
# InceptionResnetV2 may be too complex to freeze/unfreeze layers due to shear size of model
# attempt on smaller transfer learning model such as VGG16 before trying on this model.

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.utils import to_categorical

IMG_DIM = (350, 350)

os.chdir('E:/DataForCapstone')
train_files = glob.glob('Train/*')
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [re.split(r'(\d+)', fn.split('\\')[1])[0] for fn in train_files]

validation_files = glob.glob('Validate/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [re.split(r'(\d+)', fn.split('\\')[1])[0] for fn in validation_files]

print('Train dataset shape:', train_imgs.shape,
      '\tValidation dataset shape:', validation_imgs.shape)


train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled  = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

print(train_imgs[0].shape)
array_to_img(train_imgs[0])


# batch_size = 75
num_classes = 8
# epochs = 40
input_shape = (350, 350, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)

print(train_labels[700:705], train_labels_enc[700:705])

train_labels_enc = to_categorical(train_labels_enc, num_classes)
validation_labels_enc = to_categorical(validation_labels_enc, num_classes)

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)

from keras.applications import inception_resnet_v2
from keras.models import Model
import keras

inceptionResNetv2 = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

output = inceptionResNetv2.layers[-1].output
output = keras.layers.Flatten()(output)
inception_resnet_v2_model = Model(inceptionResNetv2.input, output)

inception_resnet_v2_model.trainable = False
for layer in inception_resnet_v2_model.layers:
    layer.trainable = False

import pandas as pd

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in inception_resnet_v2_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])


def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features


train_features_inception_resnet = get_bottleneck_features(inception_resnet_v2_model, train_imgs_scaled)
validation_features_inception_resnet = get_bottleneck_features(inception_resnet_v2_model, validation_imgs_scaled)

print('Train Bottleneck Features:', train_features_inception_resnet.shape,
      '\tValidation Bottleneck Features:', validation_features_inception_resnet.shape)



from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(inception_resnet_v2_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

model.summary()



history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=120,
                              validation_data=val_generator, validation_steps=50,
                              verbose=1)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic model with transfer and ImageAug CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,121))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 121, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 121, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.show()

model.save('simple_zoo_imageAug_with_Transfer_learning2_FullRun.h5')