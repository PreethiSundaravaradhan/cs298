import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras_metrics as km
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt


def get_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(3, 100, 300)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)


        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(9))
        model.add(Activation('sigmoid'))
        return model


def build_models(weights_file):
        model = get_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['mean_squared_error', 'accuracy'])


        batch_size = 16

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
                '/Users/preethi/Allclass/297/cnn_data/s_pattern_images/',  # this is the target directory
                target_size=(100, 300),  # all images will be resized to 150x150
                batch_size=batch_size,
                class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels #categorical


        model.fit_generator(
                train_generator,
                steps_per_epoch=50,
                epochs=3)

        model.save_weights(weights_file)  # always save your weights after training or during training


def rebuild_model(weights_file):
        rebuilt_model = get_model()
        rebuilt_model.load_weights(weights_file)#'first_try.h5')
        return rebuilt_model


def try_roc_sklearn(y_pred, y_true):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = 9
        lw = 2
        # Compute macro-average ROC curve and ROC area
        for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_pred[:, i], y_true[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])


        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        roc_auc[i] = auc(fpr[i], tpr[i])


        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_pred.ravel(), y_true.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Plot all ROC curves
        plt.figure()

        colors = ['aqua', 'darkorange', 'cornflowerblue','r','b','g','c','y','m']
        for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of user {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for Unique User Patterns(Acceleration)')
        plt.legend(loc="lower right")
        plt.show()


def try_predict(weights):
        # build_models()
        model = rebuild_model(weights_file=weights)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        test_datagen = test_datagen.flow_from_directory(
               '/Users/preethi/Allclass/297/data_validation_unique/',  # this is the target directory
                target_size=(100, 300),  # all images will be resized to 150x150
                batch_size=16,
                shuffle=False,
                class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

        predictions = model.predict_generator(test_datagen)
        print(predictions)

        y_pred = np.zeros((39, 9), dtype=np.int8)

        r_num = 0
        for row in predictions:
                max_idx = np.argmax(row)
                y_pred[r_num][max_idx] = 1
                r_num += 1
        #y_pred = np.amax(predictions, axis=1)

        y_true_h = test_datagen.classes

        y_true = np.zeros((y_true_h.size, int(y_true_h.max()) + 1),dtype=np.int8)
        y_true[np.arange(y_true_h.size), y_true_h] = 1

        try_roc_sklearn(y_pred, y_true)


if __name__ == "__main__":
       # build_models('cnn_common_2class3.h5')
        #s_pattern_model = rebuild_model('cnn_common_pattern.h5')
        try_predict('first_try.h5')
        #try_predict('cnn_.h5')