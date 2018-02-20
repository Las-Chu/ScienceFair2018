import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils.np_utils import to_categorical

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set this to -1 for cpu

import global_defs as gf
import data_generator as dg
import crop_images as image_cropping

to_be_trained = False

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(gf.gt_nb_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

if (to_be_trained):

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # data for trainingand validation
    # Split the lines into training and validation samples (20% for validation set)
    train_samples, validation_samples = dg.train_test_split(dg.read_gt_data(13356), test_size=0.2)
    print('train samples len', len(train_samples))

    # Set the traing and validation data generators
    batch_size = 32
    train_generator = dg.generator(train_samples, batch_size=batch_size)
    validation_generator = dg.generator(validation_samples, batch_size=batch_size)

    # train the model on the new data for a few epochs
    history_object = model.fit_generator(train_generator, steps_per_epoch=
                len(train_samples)/batch_size, validation_data=validation_generator,
                validation_steps=len(validation_samples)/batch_size, epochs=5, verbose=2)

    #model.fit_generator(X_train, Y_train, epochs=50,shuffle=True, verbose=2)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:

    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first N layers (249, 184) and unfreeze the rest:
    freeze_layers_boundary = 184
    for layer in model.layers[:freeze_layers_boundary]:
       layer.trainable = False
    for layer in model.layers[freeze_layers_boundary:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    #model.fit_generator(...)

    history_object = model.fit_generator(train_generator, steps_per_epoch=
                len(train_samples)/batch_size, validation_data=validation_generator,
                validation_steps=len(validation_samples)/batch_size, epochs=20, verbose=2)

    # Save the model to be used in testing the autonomous mode for the car
    model.save(gf.project_name+'.h5')

    # plot the training and validation loss for each epoch
    '''
    import matplotlib.pyplot as plt
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.plot(history_object.history['val_acc'])
    plt.title('Model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    '''

else:
    # Predict the image feature
    test_file = '/home/ramesh/Data/ISIC-2017_Test_v2_Data/'+ 'ISIC_0015051.jpg'
    model.load_weights(gf.project_name+'.h5')
    img = image_cropping.ret_proc_image(test_file, [229, 229])
    #img = image.load_img(gf.gt_train_val_location+'/ISIC_0016071.jpg', target_size=(229, 229))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    #print('Predicted:', decode_predictions(preds, top=3)[0])
    print(preds)

    #categorical_labels = to_categorical(preds, num_classes=None)
    #print(categorical_labels)
# plot the training and validation loss for each epoch
'''
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.plot(history_object.history['val_acc'])
plt.title('Model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
