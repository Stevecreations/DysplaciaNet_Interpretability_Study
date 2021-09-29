import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import keras
from keras import models
from keras.preprocessing import image
import sklearn
from sklearn.manifold import TSNE


seed = 10

np.random.seed(seed)

image_folder_path = ("C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\Server_Images")
#image_folder_path=("C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\Server_Images_Masked")

image_name_list = ["TD_BNE_2249981.jpg","TD_BNE_2256328.jpg","TD_SNE_741897.jpg",
                   "TN_SNE_118039.jpg","TN_SNE_14872661.jpg","TN_SNE_14872673.jpg",
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

image_class_type_list = [0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    2.0, 2.0, 2.0, 2.0, 2.0,
                    3.0, 3.0, 3.0, 3.0, 3.0
                    ]

image_score_list = [7,6,5,
                    -2,0,0,
                    4,5,6,3,7,3,3,5,5,5,6,6,6,7,7,6,7,2,6,6,
                    -3,-1,-3,-4,-1,-1,-5,-4,-5,-3,-3,-3,-3,-3,-4,-1,-3,-3,-4,-5,
                    7,-3,3,0,6,
                    1,0,-3,-3,-2
                    ]


image_cytoplasm_list = [3,3,3,
                        -1,-1,-1,
                        1,3,2,-1,3,3,1,3,2,2,3,3,3,3,3,3,3,3,3,3,
                        -1,-2,-2,-2,-2,-3,-2,-2,-3,-2,-2,-2,-2,-1,-2,-2,-2,-2,-2,-3,
                        3,-1,1,-1,2,
                        -1,-1,-2,-1,-2]

image_nucleus_list =[3,2,2,
                     -1,1,1,
                     2,1,3,3,3,-1,1,2,2,2,2,3,2,3,3,3,3,-1,3,3,
                     -2,1,-1,-2,1,2,-3,-2,-2,-1,-1,-1,-1,-2,-2,1,-1,-1,-2,-3,
                     3,-3,1,1,3,
                     1,1,-1,-2,-1]



def load_model():
    # load model
    model_path = 'model/final_model.json'
    weigths_path = 'model/final_model_weights.h5'

    with open(model_path) as json_file:
        json_config = json_file.read()
    dysplacianet_model = keras.models.model_from_json(json_config)

    dysplacianet_model.load_weights(weigths_path)
    # dysplacianet_model.add(Dense(2, activation='softmax', name='visualized_layer'))

    print(dysplacianet_model.summary())
    compilation = dysplacianet_model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return dysplacianet_model
def path_to_image_tensor(path, img_height, img_width):
    img = image.load_img(path, target_size=(img_height, img_width))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor




def convolution_4_output(model, image_path, class_index):
    img_height = 299
    img_width = 299
    img_tensor_in = path_to_image_tensor(image_path, img_height, img_width)
    layer_outputs = [layer.output for layer in model.layers[:12]]  # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(img_tensor_in)  # Returns a list of five Numpy arrays: one array per layer activation

    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot
    #print(layer_names[11])
    #print(activations[11].shape)
    flattened=activations[11].flatten()
    #print(flattened.shape)

    return flattened

dysplacianet_model=load_model()


output=[None]*len(image_name_list)
for i in range(len(image_name_list)):
    current_output=convolution_4_output(dysplacianet_model, image_folder_path+ '\\' + image_name_list[i], 1.0)
    output[i]=current_output

tsne = TSNE(n_components=2).fit_transform(output)

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):

    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)



#colors_per_class=[0.0,1.0]
#for label in colors_per_class:

#    # find the samples of the current class in the data
#    indices = [i for i, l in enumerate(image_class_list) if l == label]
#    # extract the coordinates of the points of this class only
#    current_tx = np.take(tx, indices)
#    current_ty = np.take(ty, indices)
#    # convert the class color to matplotlib format
#    #color = np.array(colors_per_class[label], dtype=np.int)

#    # add a scatter plot with the corresponding color and label
#    ax.scatter(current_tx, current_ty, label=label)

# build a legend using the labels we set previously

#ax.legend(loc='best')
# finally, show the plot
#plt.show()


fig,ax=plt.subplots(1,4)

markers = ["$-6$" , "$-5$" , "$-4$" , "$-3$" , "$-2$" , "$-1$", "$+0$", "$+1$" , "$+2$" , "$+3$" , "$+4$" , "$+5$", "$+6$", "$+7$"]
colors = ['r','g','b','c']
legend_elements=[Patch(facecolor='r',edgecolor='r', label='Dysplastic'),
                 Patch(facecolor='g',edgecolor='g', label='Normal'),
                 Patch(facecolor='b',edgecolor='b', label='Wrong Dysplastic'),
                 Patch(facecolor='c',edgecolor='c', label='Wrong Normal')]

for i in range(len(tx)):
    current_tx = tx[i]
    current_ty = ty[i]
    ax[0].scatter(current_tx, current_ty,c=colors[int(image_class_type_list[i])], marker=markers[image_score_list[i]+6],label = int(image_class_type_list[i]),s=100)

#ax[0].legend(handles=legend_elements,loc='best')
ax[0].set_title('Total Score')

for i in range(len(tx)):
    current_tx = tx[i]
    current_ty = ty[i]
    ax[1].scatter(current_tx, current_ty,c=colors[int(image_class_type_list[i])], marker=markers[image_cytoplasm_list[i]+6],label = int(image_class_type_list[i]),s=100)

#ax[1].legend(handles=legend_elements,loc='best')
ax[1].set_title('Cytoplasm Score')

for i in range(len(tx)):
    current_tx = tx[i]
    current_ty = ty[i]
    ax[2].scatter(current_tx, current_ty,c=colors[int(image_class_type_list[i])], marker=markers[image_nucleus_list[i]+6],label = int(image_class_type_list[i]),s=100)

#ax[2].legend(handles=legend_elements,loc='best')
ax[2].set_title('Chromatin Score')
ax[3].legend(handles=legend_elements,loc='best')
# finally, show the plot
plt.show()

