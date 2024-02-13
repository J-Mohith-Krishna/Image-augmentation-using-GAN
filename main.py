import os
import numpy as np
import keras.utils as image
import matplotlib.pyplot as plt
%matplotlib inline

def load_images_from_path(path, label):
    images = []
    labels = []
    for file in os.listdir(path):
        img = image.load_img(os.path.join(path, file), target_size=(224, 224, 3))
        images.append(image.img_to_array(img))
        labels.append(label)
    return images, labels

def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)

x_train = []
y_train = []
x_test = []
y_test = []

images, labels = load_images_from_path('arctic-wildlife/train/arctic_fox', 0)
show_images(images)
x_train += images
y_train += labels

images, labels = load_images_from_path('arctic-wildlife/train/walrus', 2)
show_images(images)
x_train += images
y_train += labels

images, labels = load_images_from_path('arctic-wildlife/test/polar_bear', 1)
show_images(images)
x_test += images
y_test += labels

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input

x_train = preprocess_input(np.array(x_train))
x_test = preprocess_input(np.array(x_test))
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

from tensorflow.keras.applications import ResNet50V2
base_model = ResNet50V2(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Rescaling, RandomFlip, RandomRotation, RandomTranslation, RandomZoom

model = Sequential()
model.add(Rescaling(1./255))
model.add(RandomFlip(mode='horizontal'))
model.add(RandomTranslation(0.2, 0.2))
model.add(RandomRotation(0.2))
model.add(RandomZoom(0.2))
model.add(base_model)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=10, epochs=25)

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

y_predicted = model.predict(x_test)
mat = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
class_labels = ['arctic fox', 'polar bear', 'walrus']
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted label')
plt.ylabel('Actual label')

x = image.load_img('arcticwildlife/samples/arctic_fox/arctic_fox_140.jpeg', target_size=(224, 224))
plt.xticks([])
plt.yticks([])
plt.imshow(x)
x = image.img_to_array(x)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
predictions = model.predict(x)
for i, label in enumerate(class_labels):
    print(f'{label}: {predictions[0][i]}')

x = image.load_img('arctic-wildlife/samples/walrus/walrus_143.png', target_size=(224, 224))
plt.xticks([])
plt.yticks([])
plt.imshow(x)
x = image.img_to_array(x)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
predictions = model.predict(x)
for i, label in enumerate(class_labels):
    print(f'{label}: {predictions[0][i]}')
  
