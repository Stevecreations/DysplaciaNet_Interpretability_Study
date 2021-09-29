import tensorflow as tf
import scipy
from tensorflow import keras

from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Flatten
from keras import activations
from keras.preprocessing import image
from keras import models
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
# import pillow
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.ndimage as ndimage
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import ScoreCAM
from sklearn import preprocessing
from tf_keras_vis.utils.scores import BinaryScore
from tf_keras_vis.utils.scores import CategoricalScore

import lime
from lime import lime_image
from keras.models import load_model
from skimage.segmentation import mark_boundaries, quickshift, slic, felzenszwalb

import csv

import json
from matplotlib.path import Path
import matplotlib.patches as patches

show_plots_bool = False

# region 0. INITIALISATION
tf.keras.backend.clear_session()
# endregion
# region 1.1 LOAD TRAIN IMAGES
if False:
    test_data_dir = 'C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\3_test\\3_test'

    img_width, img_height = 299, 299
    batch_size = 1
    nb_test_samples = 1000

    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary')
# endregion
# region 1.2 LOAD SINGLE dysplastic image // normal image

dysplastic_img_path = 'C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\3_test\\3_test\\dysplastic\\SNE_1896359.jpg'
normal_img_path = 'C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\3_test\\3_test\\normal\\SNE_198143.jpg'


def path_to_image_tensor(path, img_height, img_width):
    img = image.load_img(path, target_size=(img_height, img_width))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


if False:
    dysplastic_img_tensor = path_to_image_tensor(dysplastic_img_path, img_height, img_width)
    plt.imshow(dysplastic_img_tensor[0])
    plt.title('dysplastic_cell_image')
    plt.show()

if False:
    normal_img_tensor = path_to_image_tensor(normal_img_path, img_height, img_width)
    plt.imshow(normal_img_tensor[0])
    plt.title('normal_cell_image')
    plt.show()
    print(normal_img_tensor.shape)

# endregion
# region  2.LOAD MODEL FROM FILE
model_path = 'model/final_model.json'
weigths_path = 'model/final_model_weights.h5'

with open(model_path) as json_file:
    json_config = json_file.read()
dysplacianet_model = keras.models.model_from_json(json_config)

dysplacianet_model.load_weights(weigths_path)
# dysplacianet_model.add(Dense(2, activation='softmax', name='visualized_layer'))

print(dysplacianet_model.summary())
# endregion
# region 3. COMPILE MODEL
compilation = dysplacianet_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
# endregion
if False:
    # region 4. EVALUATE MODEL
    evaluation = dysplacianet_model.evaluate(
        test_generator,
        steps=nb_test_samples,
        verbose=1)

    print(f'Test loss: {evaluation[0]} / Test accuracy: {evaluation[1]}')
    # endregion
    # region 5.1 PREDICT MODEL
    prediction = dysplacianet_model.predict(
        test_generator,
        steps=nb_test_samples,
        verbose=1)
    # endregion


# region 5.2 PREDICTION ON A SINGLE IMAGE (function)
def make_prediction(input_model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)

    x = image.img_to_array(img[0])
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = input_model.predict(images)
    if print_plot:
        print("Predicted class is:", classes)
    return (classes)


# make_prediction(dysplastic_img_tensor[0], dysplacianet_model, True)  # needs to abe a tensor
# make_prediction(normal_img_tensor[0], dysplacianet_model, True)


# endregion

# region 6.1. VISUALISE CNN FILTER (function)

def filter_visualization(model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    n_channels = 3
    n_filters = 16

    for layer in model.layers:

        if 'conv' not in layer.name:
            continue
        fig, ax = plt.subplots(n_channels, n_filters, figsize=(8, 8))
        filters, bias = layer.get_weights()
        print(layer.name, filters.shape)
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        for i in range(n_filters):
            print(i)
            f = filters[:, :, :, i]
            for j in range(3):
                print(j)
                ax[j, i].imshow(f[:, :, j], cmap='gray')
                ax[j, i].title.set_text('f:' + str(j + 1))
                ax[j, i].axis('off')
        fig.suptitle(str(layer.name))
        plt.show()
        print(layer.name)
    print(model.layers[2].get_weights())


# filter_visualization(dysplacianet_model,"1","1","1")
# endregion

# region 6. ACTIVATION MAPS CNN (function)
# https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0


def activation_maps(model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    img_tensor_in = path_to_image_tensor(image_path, img_height, img_width)
    layer_outputs = [layer.output for layer in model.layers[:12]]  # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input,
                                    outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(
        img_tensor_in)  # Returns a list of five Numpy arrays: one array per layer activation
    # activations = activation_model.predict(normal_img_tensor)

    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    images_per_row = 16
    i = 0
    plot, axs = plt.subplots(12, 1)
    axs = axs.ravel()

    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                local_mean = channel_image.mean()
                local_std = channel_image.std()
                # print(local_mean,local_std, channel_image.min(), channel_image.max())
                channel_image = (channel_image - local_mean) / local_std
                # channel_image -= local_mean  # Post-processes the feature to make it visually palatable
                # channel_image /= local_std
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size

        # axs[i].figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        axs[i].set_title(layer_name, loc='right', y=-0.05, x=0)
        axs[i].grid(False)
        axs[i].imshow(display_grid, aspect='auto', cmap='viridis')
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        i = i + 1

    local_score = make_prediction(model, image_path, class_index, print_plot)
    head, tail = os.path.split(image_path)
    plot.suptitle('IMG:' + tail + '   Score: ' + str(local_score))
    plt.axis('off')
    plot.savefig((image_path[:-4] + '_activation.jpg'), dpi=250)
    if print_plot:
        plt.show()
    plt.clf()

def activation_last_conv_maps(model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    img_tensor_in = path_to_image_tensor(image_path, img_height, img_width)
    layer_outputs = [layer.output for layer in model.layers[:12]]  # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input,
                                    outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(
        img_tensor_in)  # Returns a list of five Numpy arrays: one array per layer activation
    # activations = activation_model.predict(normal_img_tensor)

    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    images_per_row = 16
    i = 0
    plot, axs = plt.subplots(2, 1)
    axs = axs.ravel()

    print(layer_names)
    print(len(activations))

    for layer_name, layer_activation in zip(layer_names[9], activations[9]):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[
                                :, :,
                                col * images_per_row + row]
                local_mean = channel_image.mean()
                local_std = channel_image.std()
                # print(local_mean,local_std, channel_image.min(), channel_image.max())
                channel_image = (channel_image - local_mean) / local_std
                # channel_image -= local_mean  # Post-processes the feature to make it visually palatable
                # channel_image /= local_std
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size

        # axs[i].figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        axs[i].set_title(layer_name, loc='right', y=-0.05, x=0)
        axs[i].grid(False)
        axs[i].imshow(display_grid, aspect='auto', cmap='viridis')
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        i = i + 1

    local_score = make_prediction(model, image_path, class_index, print_plot)
    head, tail = os.path.split(image_path)
    plot.suptitle('IMG:' + tail + '   Score: ' + str(local_score))
    plt.axis('off')
    plot.savefig((image_path[:-4] + '_activation_last.jpg'), dpi=250)
    if print_plot:
        plt.show()
    plt.clf()


# endregion

# https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb

# region 7.5. ADDITIONAL FUNCTIONS FOR FOLLOWING FUNCTIONS
def loss(output):
    # returns data of first image
    return (output[:][0])

def score_function_normal(output):  # output shape is (batch_size, 1)
    output = output[:, 0]
    #print(output.shape)
    return output


def score_function_dysplastic(output):  # output shape is (batch_size, 1)
    output=-1*output[:, 0]
    return (output)



def model_modifier(m):
    # converts las layer to an linear activation function
    m.layers[-1].activation = tf.keras.activations.linear
    return m


def plot_result(plot_title, image_path, original_image, result_image, save_plot):
    head, tail = os.path.split(image_path)
    plt.title('IMG:' + tail + ' ' + plot_title)
    plt.imshow(original_image)
    #plt.imshow(result_image, cmap='viridis', alpha=0.7)
    plt.imshow(result_image, cmap='jet', alpha=0.4)
    plt.axis('off')
    plt.colorbar()
    plt.savefig((image_path[:-4] + '_' + plot_title + '_plot.jpg'))
    if save_plot:
        plt.show()
    # plt.close()
    plt.clf()
    save_numpy_mask(plot_title, image_path, result_image)


def save_numpy_mask(plot_title, image_path, result_image):
    # head, tail = os.path.split(image_path)
    numpydata = np.asarray(result_image)
    np.savez((image_path[:-4] + '_' + plot_title + '_mask.npz'), numpydata)


# endregion

# region 7. OCCLUSION SENSITIVITY MAPS (function)

def occlusion_maps(model, image_path, class_index, print_plot, normalize_plot=True, patch_size=10):
    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)
    class_index=int(class_index)


    sensitivity_map = np.zeros((img.shape[1], img.shape[1]))
    # Iterate the patch over the image
    for top_left_x in range(0, img.shape[1], patch_size):
        for top_left_y in range(0, img.shape[2], patch_size):
            patched_image = apply_grey_patch(img[0], top_left_x, top_left_y, patch_size)

            #plt.imshow(patched_image)
            #plt.show()
            predicted_classes=np.zeros(2)
            predicted_classes[1]=model.predict(np.array([patched_image])) #normal cell class
            predicted_classes[0]=1-predicted_classes[1] # dysplastic cell class
            # Save confidence for this specific patched image in map
            sensitivity_map[ top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] = predicted_classes[class_index]

    gaus = ndimage.gaussian_filter(sensitivity_map, sigma=3)
    if normalize_plot:
        gaus = normalize(gaus)
    if print_plot:
        plt.imshow(sensitivity_map)
        plt.show()
    plot_result(('occlusion_map_' + str(patch_size)), image_path, img[0], gaus, print_plot)
    print('Done Oclussion')

def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, 0] = 0.5 #red channel--0 black 1 white
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, 1] = 0.5 #green channel --  0 black 1 white
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, 2] = 0.5 #blue channel  -- 0 black 1 white
    #fig, axs= plt.subplots(2, 2)
    #print(patched_image.shape)
    #axs[0,0].imshow(patched_image)
    #axs[0,0].axes.xaxis.set_visible(False)
    #axs[0,0].axes.yaxis.set_visible(False)
    #axs[0,1].imshow(patched_image[:,:,0], cmap='Reds')
    ##axs[0, 1].axes.xaxis.set_visible(False)
    #axs[0, 1].axes.yaxis.set_visible(False)
    #axs[1,0].imshow(patched_image[:, :,1], cmap='Greens')
    #axs[1, 0].axes.xaxis.set_visible(False)
    #axs[1, 0].axes.yaxis.set_visible(False)
    #axs[1,1].imshow(patched_image[:, :,2], cmap='Blues')
    #axs[1, 1].axes.xaxis.set_visible(False)
    #axs[1, 13].axes.yaxis.set_visible(False)
    #plt.show()
    return patched_image

def occlusion_maps_avg(model, image_path, class_index, print_plot, normalize_plot=True, patch_size=10):
    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)
    class_index=int(class_index)


    sensitivity_map = np.zeros((img.shape[1], img.shape[1]))
    # Iterate the patch over the image
    for top_left_x in range(0, img.shape[1], patch_size):
        for top_left_y in range(0, img.shape[2], patch_size):
            patched_image = apply_average_patch(img[0], top_left_x, top_left_y, patch_size)

            #plt.imshow(patched_image)
            #plt.show()
            predicted_classes=np.zeros(2)
            predicted_classes[1]=model.predict(np.array([patched_image])) #normal cell class
            predicted_classes[0]=1-predicted_classes[1] # dysplastic cell class
            # Save confidence for this specific patched image in map
            sensitivity_map[ top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] = predicted_classes[class_index]

    gaus = ndimage.gaussian_filter(sensitivity_map, sigma=3)
    if normalize_plot:
        gaus = normalize(gaus)
    if print_plot:
        plt.imshow(sensitivity_map)
        plt.show()
    plot_result(('occlusion_map_avg_' + str(patch_size)), image_path, img[0], gaus, print_plot)
    print('Done Oclussion')

def apply_average_patch(image, top_left_x, top_left_y, patch_size):
    patched_image = np.array(image, copy=True)
    avg_0 = 0
    avg_1 = 0
    avg_2 = 0
    if top_left_y+patch_size<=299:
        i_range=patch_size
    else:
        i_range = 299-top_left_y

    if top_left_x+patch_size<=299:
        j_range=patch_size
    else:
        j_range = 299-top_left_x


    for i in range(i_range):
        for j in range(j_range):
            avg_0 = avg_0 + patched_image[top_left_y + i, top_left_x + j, 0]
            avg_1 = avg_1 + patched_image[top_left_y + i, top_left_x + j, 1]
            avg_2 = avg_2 + patched_image[top_left_y + i, top_left_x + j, 2]
    avg_0 = avg_0 / (patch_size * patch_size)
    avg_1 = avg_1 / (patch_size * patch_size)
    avg_2 = avg_2 / (patch_size * patch_size)
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, 0] = avg_0 #red channel--0 black 1 white
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, 1] = avg_1 #green channel --  0 black 1 white
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, 2] = avg_2 #blue channel  -- 0 black 1 white
    #fig, axs= plt.subplots(2, 2)
    #print(patched_image.shape)
    #axs[0,0].imshow(patched_image)
    #axs[0,1].imshow(patched_image[:,:,0], cmap='Reds')
    #axs[1,0].imshow(patched_image[:, :,1], cmap='Greens')
    #axs[1,1].imshow(patched_image[:, :,2], cmap='Blues')
    #plt.show()
    return patched_image

# endregion

# region 8. SALIENCY MAPS (functions)
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

def vanilla_saliency(model, image_path, class_index, print_plot, normalize_plot=True):
    # https: // arxiv.org / pdf / 1312.6034.pdf

    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)

    # Create Saliency object.
    saliency = Saliency(model,
                        #model_modifier=model_modifier,
                        model_modifier=ReplaceToLinear(),
                        clone=False)

    # Generate saliency map with smoothing that reduce noise by adding noise
    if class_index == 1.0:
        saliency_map = saliency(score=score_function_normal,
                                seed_input=img[0],
                                normalize_map=normalize_plot

                                )  # noise spread level.
    if class_index == 0.0:
        saliency_map = saliency(score=score_function_dysplastic,
                                seed_input=img[0],
                                normalize_map=normalize_plot
                                )  # noise spread level.

    #saliency_map = normalize(saliency_map)

    plot_result('vanilla_saliency', image_path, img[0], saliency_map[0], print_plot)
    print('Done Vanilla Saliency')

def smoothgrad_saliency(model, image_path, class_index, print_plot):
    # https://arxiv.org/pdf/1706.03825.pdf

    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)

    # Create Saliency object.
    saliency = Saliency(model,
                        # model_modifier=model_modifier,
                        model_modifier=ReplaceToLinear(),
                        clone=False)

    # Generate saliency map with smoothing that reduce noise by adding noise

    if class_index == 1.0:
        saliency_map = saliency(score_function_normal,
                                img[0],
                                smooth_samples=50,  # The number of calculating gradients iterations.
                                smooth_noise=0.150
                                )  # noise spread level.
    if class_index == 0.0:
        saliency_map = saliency(score_function_dysplastic,
                                img[0],
                                smooth_samples=50,  # The number of calculating gradients iterations.
                                smooth_noise=0.150
                                )  # noise spread level.

    # saliency_map = normalize(saliency_map)

    plot_result('smoothgrad_saliency', image_path, img[0], saliency_map[0], print_plot)
    print('Done smoothgrad Saliency')






# endregion

# region 9.GradCam
def gradcam_maps(model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)

    gradcam = Gradcam(model, model_modifier, clone=False)


    # Generate saliency map with smoothing that reduce noise by adding noise

    if class_index == 1.0:
        cam = gradcam(score_function_normal,
                      img[0],
                      penultimate_layer=-1,  # model.layers number
                      )

    if class_index == 0.0:
        cam = gradcam(score_function_dysplastic,
                      img[0],
                      penultimate_layer=-1,  # model.layers number
                      )

    cam = normalize(cam)  # not necessary
    plot_result('gradcam', image_path, img[0], cam[0], print_plot)
    print('done gradcam')

def gradcamplusplus_maps(model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)

    gradcam = GradcamPlusPlus(model, model_modifier, clone=False)

    # Generate saliency map with smoothing that reduce noise by adding noise
    if class_index == 1.0:
        cam = gradcam(score_function_normal,
                      img[0],
                      penultimate_layer=-1,  # model.layers number

                      )

    if class_index == 0.0:
        cam = gradcam(score_function_dysplastic,
                      img[0],
                      penultimate_layer=-1,  # model.layers number
                      )

    cam = normalize(cam)
    plot_result('gradcamplusplus', image_path, img[0], cam[0], print_plot)
    print('done gradcam++')
# endregion

# region 10.Scorecam
def scorecam_maps(model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)

    scorecam = ScoreCAM(model, model_modifier, clone=False)

    # Generate heatmap with Faster-ScoreCAM
    if class_index == 1.0:
        cam = scorecam(score_function_normal,
                       img[0],
                       penultimate_layer=-1,  # model.layers number
                       max_N=-1
                       )

    if class_index == 0.0:
        cam = scorecam(score_function_dysplastic,
                       img[0],
                       penultimate_layer=-1,  # model.layers number
                       max_N=-1
                       )
    cam = normalize(cam)

    plot_result('scorecam', image_path, img[0], cam[0], print_plot)

def fastscorecam_maps(model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)

    scorecam = ScoreCAM(model, model_modifier, clone=False)

    # Generate heatmap with Faster-ScoreCAM
    if class_index == 1.0:
        cam = scorecam(score_function_normal,
                       img[0],
                       penultimate_layer=-1,  # model.layers number
                       max_N=1
                       )

    if class_index == 0.0:
        cam = scorecam(score_function_dysplastic,
                       img[0],
                       penultimate_layer=-1,  # model.layers number
                       max_N=1
                       )

    cam = normalize(cam)

    plot_result('fastscorecam', image_path, img[0], cam[0], print_plot)

# endregion

# region 11. Lime

def lime_maps(model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)

    #configuration
    num_features = 10 # top superpixels number
    num_samples =700 #iterations occluding superpixels

    def segmentation_algorithm(img):
        # superpixels = slic(img_proc, n_segments=30, compactness=50, sigma=1)
        superpixels = quickshift(img.astype('double'), ratio=0.5, kernel_size=2, max_dist=150)
        # superpixels = quickshift(image, ratio=0.2, kernel_size=6, max_dist=200)
        # superpixels = felzenszwalb(img, scale=200, sigma=1, min_size=200)
        return superpixels

    superpixels=segmentation_algorithm(img[0])
    num_superpixels = np.unique(superpixels).shape[0]
    print('Number of superpixels: ' + str(num_superpixels))
    explainer = lime_image.LimeImageExplainer(verbose=True, feature_selection='lasso_path', random_state=232)
    explanation = explainer.explain_instance(img[0], model.predict, top_labels=1, hide_color=None,
                                             num_samples=num_samples,
                                             num_features=num_superpixels,
                                             segmentation_fn=segmentation_algorithm,
                                             random_seed=10)


    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False,negative_only=False,
                                                num_features=num_features, hide_rest=True)
    if print_plot:
        fig, ax = plt.subplots(1, 3, figsize=(10,10))
        ax[0].imshow(img[0])
        ax[0].title.set_text('Explanation Score:' + str(explanation.score))
        ax[1].imshow(img[0])
        ax[1].imshow(mask, cmap='Set3', alpha=0.7)
        ax[1].title.set_text('Nº features:'+ str(num_features))
        ax[2].imshow(mark_boundaries(img[0], superpixels))
        ax[2].title.set_text('Nº superpixels:'+ str(num_superpixels))
        plt.show()

    plot_result('Lime', image_path, img[0], mask, print_plot)
    plot_result('Lime_segmentation', image_path, img[0], mark_boundaries(img[0], superpixels), print_plot)

def lime_V2_maps(model, image_path, class_index, print_plot):
    img_height = 299
    img_width = 299
    img = path_to_image_tensor(image_path, img_height, img_width)

    superpixels = quickshift(img[0].astype('double'), ratio=0.2, kernel_size=4, max_dist=600)
    num_superpixels = np.unique(superpixels).shape[0]
    print('Number of superpixels: ' + str(num_superpixels))

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].imshow(mark_boundaries(img[0], superpixels))

    # Generate perturbations
    num_perturb = 500
    perturbations = np.random.binomial(1, 0.80, size=(num_perturb, num_superpixels))

    # Create function to apply perturbations to images
    import copy
    def perturb_image(img, perturbation, segments):
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        #print(mask.shape)

        #plt.imshow(mask)
        #plt.show()
        #for active in active_pixels:
        #    mask[segments != active] = 0.3
        ##print(mask)
        #plt.imshow(mask, cmap='Greys')
        #plt.show()
        for active in active_pixels:
            mask[segments == active] = 1
        #print(mask)
        #plt.imshow(mask, cmap='Greys')
        #plt.show()
        perturbed_image = copy.deepcopy(img)
        perturbed_image = perturbed_image * mask[:, :, np.newaxis]
        return perturbed_image

    # Show example of perturbations
    print('pertubations' + str(perturbations[0]))
    ax[0, 1].imshow(perturb_image(img[0], perturbations[0], superpixels))

    predictions = []
    for pert in perturbations:
        perturbed_img = perturb_image(img[0], pert, superpixels)

        pred = model.predict(perturbed_img[np.newaxis, :, :, :])
        if class_index == 1.0:
            predictions.append(pred)

        if class_index == 0.0:
            predictions.append(1.0 - pred)

        #plt.imshow(perturbed_img)
        #plt.title(str(pred))
        #plt.show()

    predictions = np.array(predictions)
    # print(predictions[:,0,0])
    print('predictions_shape:' + str(predictions.shape))

    # Compute distances to original image
    import sklearn.metrics
    original_image = np.ones(num_superpixels)[np.newaxis, :]  # Perturbation with all superpixels enabled
    distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()
    print('distance_shape:' + str(distances.shape))

    # Transform distances to a value between 0 an 1 (weights) using a kernel function
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function
    print('wheights_shape:' + str(weights.shape))

    # Estimate linear model
    from sklearn.linear_model import LinearRegression
    # class_to_explain = top_pred_classes[0]  # Labrador class
    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions[:, :, 0], sample_weight=weights)

    coeff = simpler_model.coef_[0]
    print(predictions[:, :, 0])
    print(coeff)
    # Use coefficients from linear model to extract top features
    num_top_features = 5
    top_features = np.argsort(coeff)[-num_top_features:]
    print(top_features)

    # Show only the superpixels corresponding to the top features
    mask = np.zeros(num_superpixels)
    mask[top_features] = True  # Activate top superpixels
    ax[1, 0].imshow(perturb_image(img[0], mask, superpixels))
    plt.show()
    print('Done')


# lime_V2_maps(dysplacianet_model, "C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\Server_Images" + '\\' + "TD_BNE_2256328.jpg", 0, show_plots_bool)

# endregion

# region 12. ANNOTATION APPLICATION DATA

def obtain_annotation_data(model, image_path, class_index, print_plot):
    json_mask_data = os.path.join(image_path[:-4] + "_json.txt")  # file of document to open

    if os.path.isfile(json_mask_data):  # file with notation shapes exists
        with open(json_mask_data) as json_file:
            masks_data = json.load(json_file)
        image_saved_data = masks_data

        original_image_x = 360
        original_image_y = 363
        analysis_image_x = 299
        analysis_image_y = 299

        # obtain inputed data
        cell_condition = image_saved_data["cellstatus"]

        existing_colors=[None]*(len(image_saved_data["shapes"]))
        for i in range(len(image_saved_data["shapes"])):

           existing_colors[i] = image_saved_data["shapes"][i]["line"]["color"]

        print(existing_colors)
        if '#cb4335' in existing_colors:
            print('Transparent')
            cytoplasm_condition = 'Transparent'
            cytoplasm_score = image_saved_data["priorities"][0][0]
        elif '#e67e22' in existing_colors:
            print('Granular')
            cytoplasm_condition = 'Granulated'
            cytoplasm_score = (-1) * (image_saved_data["priorities"][0][0])
        else:
            print('Fail cyto')
            cytoplasm_condition = 'Granulated'
            cytoplasm_score = (-1) * (image_saved_data["priorities"][0][0])

        if '#5dade2' in existing_colors:
            print('Heterogenic')
            chroma_condition = 'Heterogenic'
            chroma_score =  (image_saved_data["priorities"][0][1])
        elif '#76d7c4' in existing_colors:
            print('Homogenic')
            chroma_condition = 'Homogenic'
            chroma_score = (-1) *(image_saved_data["priorities"][0][1])
        else:
            print('Fail chroma')
            chroma_condition = 'Homogenic'
            chroma_score = image_saved_data["priorities"][0][1]


        #if cell_condition == 'DYSPLASTIC':
        #    cytoplasm_condition = 'Transparent'
        #    cytoplasm_score = image_saved_data["priorities"][0][0]
        #else:
        #    cytoplasm_condition = 'Granulated'
        #    cytoplasm_score = (-1) * (image_saved_data["priorities"][0][0])

        #if cell_condition == 'DYSPLASTIC':
        #    chroma_condition = 'Heterogenic'
        #    chroma_score = (-1) * (image_saved_data["priorities"][0][1])
        #else:
        #    chroma_condition = 'Homogenic'
        #    chroma_score = image_saved_data["priorities"][0][1]

        lobes_condition = image_saved_data["lobes"]
        lobes_score = image_saved_data["priorities"][0][2]

        # initialisation
        fig, ax1 = plt.subplots(figsize=(10, 10))
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax1.set_xlim(0, original_image_x)
        ax1.set_ylim(0, original_image_y)
        fig.gca().invert_yaxis()
        ax2.set_xlim(0, original_image_x)
        ax2.set_ylim(0, original_image_y)
        fig2.gca().invert_yaxis()

        for i in range(0, len(image_saved_data["shapes"])):

            plotly_path = image_saved_data["shapes"][i]["path"]
            plotly_color = image_saved_data["shapes"][i]["line"]["color"]
            plotly_width = image_saved_data["shapes"][i]["line"]["width"]
            # print(plotly_color)

            codes = []
            verts = []

            path_string = plotly_path.split("L")
            for i in range(0, len(path_string)):
                # print(path_string[i])
                position = []
                if "M" in path_string[i]:
                    path_string[i] = path_string[i].replace('M', '')
                    x_coor, y_coor = path_string[i].split(",")
                    x_coor = float(x_coor)
                    y_coor = float(y_coor)
                    position.insert(0, x_coor)
                    position.insert(1, y_coor)
                    verts.insert(i, tuple(position))
                    codes.insert(i, np.uint8(1))

                elif "Z" in path_string[i]:
                    # print(path_string[i])
                    path_string[i] = path_string[i].replace('Z', '')
                    x_coor, y_coor = path_string[i].split(",")
                    x_coor = float(x_coor)
                    y_coor = float(y_coor)
                    position.insert(0, x_coor)
                    position.insert(1, y_coor)
                    verts.insert(i, tuple(position))
                    codes.insert(i, np.uint8(79))

                else:
                    # "L"
                    x_coor, y_coor = path_string[i].split(",")
                    x_coor = float(x_coor)
                    y_coor = float(y_coor)
                    position.insert(0, (x_coor))
                    position.insert(1, (y_coor))
                    verts.insert(i, tuple(position))
                    codes.insert(i, np.uint8(2))

            path = Path(verts, codes)

            patch = patches.PathPatch(path, edgecolor=plotly_color, facecolor=plotly_color, lw=plotly_width)
            if plotly_color == '#cb4335' or plotly_color == '#e67e22':
                ax1.add_patch(patch)
            if plotly_color == '#5dade2' or plotly_color == '#76d7c4':
                ax2.add_patch(patch)

        ax1.axis('off')
        ax1.margins(0)
        fig.tight_layout(pad=0)
        fig.gca().set_aspect('equal', adjustable='box')
        fig.canvas.draw()
        ax2.axis('off')
        ax2.margins(0)
        fig2.tight_layout(pad=0)
        fig2.gca().set_aspect('equal', adjustable='box')
        fig2.canvas.draw()

        fig.savefig(image_path[:-4] + '_annotation_cytoplasm_mask.png', bbox_inches='tight',
                    format='png')
        fig2.savefig(image_path[:-4] + '_annotation_nucleus_mask.png', bbox_inches='tight',
                     format='png')

        im = cv2.imread(image_path[:-4] + '_annotation_nucleus_mask.png')
        im = cv2.resize(im, dsize=(analysis_image_x, analysis_image_y), interpolation=cv2.INTER_CUBIC)
        # print(type(im), im.shape, np.unique(im, return_counts=True), np.unique(im))
        np.savez((image_path[:-4] + '_annotation_nucleus_mask.npz'), im)

        im = cv2.imread(image_path[:-4] + '_annotation_cytoplasm_mask.png')
        im = cv2.resize(im, dsize=(analysis_image_x, analysis_image_y), interpolation=cv2.INTER_CUBIC)
        # print(type(im), im.shape, np.unique(im, return_counts=True), np.unique(im))
        np.savez((image_path[:-4] + '_annotation_cytoplasm_mask.npz'), im)

        if print_plot:# print plot
            plt.show()
        fig.clf()
        plt.cla()
        plt.close()
        fig2.clf()
        plt.cla()
        plt.close()

        return (cell_condition[0], cytoplasm_condition, cytoplasm_score, chroma_condition, chroma_score, lobes_condition[0],
                lobes_score)

    else:
        print('no_annotation_file')
        return ('n/a', 'n/a', 0, 'n/a', 0, 'n/a', 0)


# endregion

# region 13. COMPARE MASK DATA
def mask_data(image_path, method, print_plot):
    method_mask_path = os.path.join(image_path[:-4] + '_' + method + "_mask.npz")  # file of document to open
    nucleus_mask_path = os.path.join(image_path[:-4] + '_annotation_nucleus_mask.npz')  # file of document to open
    cytoplasm_mask_path = os.path.join(image_path[:-4] + '_annotation_cytoplasm_mask.npz')  # file of document to open

    if os.path.isfile(method_mask_path) and os.path.isfile(nucleus_mask_path) and os.path.isfile(cytoplasm_mask_path):

        method_mask_data = np.load(method_mask_path)
        nucleus_mask_data = np.load(nucleus_mask_path)
        cytoplasm_mask_data = np.load(cytoplasm_mask_path)

        # for k in  method_mask_data.iterkeys():
        #    print(k)

        # print(method_mask_data['arr_0'], nucleus_mask_data['arr_0'], cytoplasm_mask_data['arr_0'] )
        method_mask_data = method_mask_data['arr_0']
        nucleus_mask_data = nucleus_mask_data['arr_0']
        cytoplasm_mask_data = cytoplasm_mask_data['arr_0']

        # print(method_mask_data.shape, nucleus_mask_data.shape, cytoplasm_mask_data.shape)

        def rgb2gray(rgb):

            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

            return gray

        nucleus_mask_data = rgb2gray(nucleus_mask_data)
        cytoplasm_mask_data = rgb2gray(cytoplasm_mask_data)

        nucleus_mask_data = np.where(nucleus_mask_data > 250, 0, 1)
        cytoplasm_mask_data = np.where(cytoplasm_mask_data > 250, 0, 1)
        cytoplasm_mask_data = cytoplasm_mask_data - nucleus_mask_data
        cytoplasm_mask_data = np.where(cytoplasm_mask_data > 0.5, 1, 0)

        nucleus_method_img = method_mask_data * nucleus_mask_data
        cytoplasm_method_img = method_mask_data * cytoplasm_mask_data

        nucleus = np.where(nucleus_mask_data == 0)
        nucleus_method = method_mask_data[nucleus]

        #print(nucleus_method_img.shape, np.mean(nucleus_method), np.max(nucleus_method), np.min(nucleus_method))

        cytoplasm = np.where(cytoplasm_mask_data == 0)
        cytoplasm_method = method_mask_data[cytoplasm]
        #print(cytoplasm_method_img.shape, np.mean(cytoplasm_method), np.max(cytoplasm_method), np.min(cytoplasm_method))

        if print_plot:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            ax1.imshow(nucleus_method_img)
            ax2.imshow(cytoplasm_method_img)
            ax3.imshow(nucleus_mask_data, cmap='gray')
            ax4.imshow(cytoplasm_mask_data, cmap='gray')

            plt.show()
        print('extract mask: ok')
        return (np.mean(nucleus_method), np.max(nucleus_method), np.min(nucleus_method), np.mean(cytoplasm_method),
                np.max(cytoplasm_method), np.min(cytoplasm_method))

    else:
        print('unknown method or mask')
        return (0, 0, 0, 0, 0, 0)



# endregion
# region 20 RUN ANALYSIS

image_folder_path = ("C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\Server_Images")
image_folder_path=("C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\Server_Images_Masked")
#image_folder_path = ("C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\final_images")
# image_folder_path=("C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\Annotation_Masked")
# image_name_list=["TD_BNE_2249981.jpg"]

image_name_list = ["TD_BNE_2249981.jpg",
                   "TD_BNE_2256328.jpg",
                   "TD_SNE_741897.jpg",
                   "TN_SNE_118039.jpg",
                   "TN_SNE_14872661.jpg",
                   "TN_SNE_14872673.jpg",
                   "ZD_BNE_2433069.jpg",
                   "ZD_BNE_2433092.jpg",
                   "ZD_BNE_2433255.jpg",
                   "ZD_BNE_2433280.jpg",
                   "ZD_BNE_2759813.jpg",
                   "ZD_SNE_2433225.jpg",
                   "ZD_SNE_2543049.jpg",
                   "ZD_SNE_2543060.jpg",
                   "ZD_SNE_2543086.jpg",
                   "ZD_SNE_2759718.jpg",
                   "ZD_SNE_2759795.jpg",
                   "ZD_SNE_6643852.jpg",
                   "ZD_SNE_6643874.jpg",
                   "ZD_SNE_6643949.jpg",
                   "ZD_SNE_6700392.jpg",
                   "ZD_SNE_6700394.jpg",
                   "ZD_SNE_6700404.jpg",
                   "ZD_SNE_6700428.jpg",
                   "ZD_SNE_6700434.jpg",
                   "ZD_SNE_6700475.jpg",
                   "ZN_SNE_4198926.jpg",
                   "ZN_SNE_4198932.jpg",
                   "ZN_SNE_4198946.jpg",
                   "ZN_SNE_4198957.jpg",
                   "ZN_SNE_4198961.jpg",
                   "ZN_SNE_4198990.jpg",
                   "ZN_SNE_4199013.jpg",
                   "ZN_SNE_4199046.jpg",
                   "ZN_SNE_4223869.jpg",
                   "ZN_SNE_4223913.jpg",
                   "ZN_SNE_5456732.jpg",
                   "ZN_SNE_5456876.jpg",
                   "ZN_SNE_5456885.jpg",
                   "ZN_SNE_5456900.jpg",
                   "ZN_SNE_5456910.jpg",
                   "ZN_SNE_5456916.jpg",
                   "ZN_SNE_5456924.jpg",
                   "ZN_SNE_5456933.jpg",
                   "ZN_SNE_5456968.jpg",
                   "ZN_SNE_5456971.jpg",
                   "WD_BNE_2426191.jpg",
                   "WD_SNE_2288170.jpg",
                   "WD_SNE_2426196.jpg",
                   "WD_SNE_2426212.jpg",
                   "WD_SNE_2683958.jpg",
                   "WN_BNE_3328133.jpg",
                   "WN_SNE_3328147.jpg",
                   "WN_SNE_5500961.jpg",
                   "WN_SNE_5500993.jpg",
                   "WN_SNE_5501052.jpg"
                   ]

image_class_list = [0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0, 1.0
                    ]

with open(image_folder_path + '\\' + 'Prediction.csv', 'w', newline='') as csvfile:
    fieldnames = ['image', 'class', 'score', 'prediction', 'pathologist_state', 'cytoplasm_state', 'cytoplasm_score',
                  'nucleus_state', 'nucleus_score', 'lobe_state', 'lobe_score',
                 ]
    score_file = csv.DictWriter(csvfile, fieldnames=fieldnames)
    score_file.writeheader()

    for image_name, image_class in zip(image_name_list, image_class_list):
        print(image_name)
        prediction = make_prediction(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, True)
        annotation = obtain_annotation_data(dysplacianet_model, image_folder_path + '\\' + image_name, image_class,
                                            False)


        # activation_maps(dysplacianet_model,image_folder_path + '\\' + image_name, image_class, show_plots_bool)
        #activation_last_conv_maps(dysplacianet_model,image_folder_path + '\\' + image_name, image_class, show_plots_bool)
        #smoothgrad_saliency(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool)
        #smoothgrad_data = mask_data(image_folder_path + '\\' + image_name, 'smoothgrad_saliency', False)
        #vanilla_saliency(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool)
        #vanilla_data = mask_data(image_folder_path + '\\' + image_name, 'vanilla_saliency', False)
        #gradcam_maps(dysplacianet_model, image_folder_path +  '\\' +image_name,1.0 , show_plots_bool)
        #gradcamplusplus_maps(dysplacianet_model, image_folder_path + '\\' + image_name,1.0, show_plots_bool)
        #fastscorecam_maps(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool)
        #scorecam_maps(dysplacianet_model, image_folder_path + '\\' + image_name, 1.0, show_plots_bool)
        # activation_maps(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool)
        #lime_maps(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool)
        #lime_V2_maps(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool)
        #occlusion_maps(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False,patch_size= 2)
        #occlusion_maps(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False,patch_size= 5)
        #occlusion_maps(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False,patch_size= 10)
        #occlusion_maps(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False,patch_size= 20)
        #occlusion_maps(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False, patch_size=40)
        #occlusion_maps_avg(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False,patch_size= 2)
        #occlusion_maps_avg(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False,patch_size= 5)
        #occlusion_maps_avg(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False,patch_size= 10)
        #occlusion_maps_avg(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False,patch_size= 20)
        #occlusion_maps_avg(dysplacianet_model, image_folder_path + '\\' + image_name, image_class, show_plots_bool,normalize_plot=False, patch_size=40)


        score_file.writerow(
            {'image': image_name, 'class': image_class, 'score': prediction[0][0], 'pathologist_state': annotation[0],
             'cytoplasm_state': annotation[1], 'cytoplasm_score': annotation[2],
             'nucleus_state': annotation[3], 'nucleus_score': annotation[4],
             'lobe_state': annotation[5], 'lobe_score': annotation[6],
             })

# endregion
