import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras

from keras.preprocessing import image
import cv2
import csv

overlay=True
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

image_name_list_normal = ["TN_SNE_118039.jpg","TN_SNE_14872661.jpg","TN_SNE_14872673.jpg",

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
                   ]
image_class_list_normal = [
                    1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   ]

image_name_list_dysplastic = ["TD_BNE_2249981.jpg","TD_BNE_2256328.jpg","TD_SNE_741897.jpg",
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

                   ]
image_class_list_dysplastic = [0.0, 0.0, 0.0,

                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

                    ]

image_name_list_dysplastic_m = [
                   "WD_BNE_2426191.jpg",
                   "WD_SNE_2288170.jpg",
                   "WD_SNE_2426196.jpg",
                   "WD_SNE_2426212.jpg",
                   "WD_SNE_2683958.jpg",
                   #"WN_BNE_3328133.jpg",
                   #"WN_SNE_3328147.jpg",
                   #"WN_SNE_5500961.jpg",
                   #"WN_SNE_5500993.jpg",
                   #"WN_SNE_5501052.jpg"
                   ]
image_class_list_dysplastic_m = [
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    #1.0, 1.0, 1.0, 1.0, 1.0
                    ]

image_name_list_normal_m = [
                   #"WD_BNE_2426191.jpg",
                   #"WD_SNE_2288170.jpg",
                   #"WD_SNE_2426196.jpg",
                   #"WD_SNE_2426212.jpg",
                   #"WD_SNE_2683958.jpg",
                   "WN_BNE_3328133.jpg",
                   "WN_SNE_3328147.jpg",
                   "WN_SNE_5500961.jpg",
                   "WN_SNE_5500993.jpg",
                   "WN_SNE_5501052.jpg"
                   ]
image_class_list_normal_m = [
                    #0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0, 1.0
                    ]

extension = [

    #"_annotation_cytoplasm_mask.npz",
    #"_annotation_nucleus_mask.npz",
    #"_test.npz",
    #"_occlusion_map_5_mask.npz",
    #"_occlusion_map_10_mask.npz",
    #"_occlusion_map_20_mask.npz",
    #"_occlusion_map_40_mask.npz",
    "_occlusion_map_avg_5_mask.npz",
    "_occlusion_map_avg_10_mask.npz",
    "_occlusion_map_avg_20_mask.npz",
    "_occlusion_map_avg_40_mask.npz",
    #"_scorecam_mask.npz",
    #"_fastscorecam_mask.npz",
    #"_gradcam_mask.npz",
    #"_gradcamplusplus_mask.npz",
    #"_lime_mask.npz",
    #"_vanilla_saliency_mask.npz",
    #"_smoothgrad_saliency_mask.npz",
]

#file_path = ("C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\Annotation")

overlay=True

def path_to_image_tensor(path, img_height, img_width):
    img = image.load_img(path, target_size=(img_height, img_width))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

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


def obtain_distribution(file_path, image_name_list, image_class_list, extension):

    file_extension = []
    file_extension = extension.copy()
    dysplacianet_model=load_model()
    result=np.zeros(shape=(len(extension),len(image_name_list),2))

    with open(file_path + '\\' + 'occlusion_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'class', 'score', '5_min','5_max','5_std', '10_min','10_max','10_std', '20_min','20_max','20_std', '40_min','40_max','40_std',]
        score_file = csv.DictWriter(csvfile, fieldnames=fieldnames)
        score_file.writeheader()

        for j in range(len(image_name_list)):
            image_name = image_name_list[j]
            image_class = image_class_list[j]
            prediction = make_prediction(dysplacianet_model, file_path + "\\" + image_name, image_class, False)
            prediction =prediction [0][0]


            if os.path.isfile(file_path + "\\" +image_name[:-4] + file_extension[0]):
                data = np.load(file_path + "\\" + image_name[:-4] + file_extension[0])
                max5 = np.max(data['arr_0'])
                min5 = np.min(data['arr_0'])
                std5 = np.std(data['arr_0'])

            if os.path.isfile(file_path + "\\" + image_name[:-4]+ file_extension[1]):
                data = np.load(file_path + "\\" + image_name[:-4]+file_extension[1])
                max10 = np.max(data['arr_0'])
                min10 = np.min(data['arr_0'])
                std10 = np.std(data['arr_0'])

            if os.path.isfile(file_path + "\\" + image_name[:-4]+file_extension[2]):
                data = np.load(file_path + "\\" + image_name[:-4]+file_extension[2])
                max20 = np.max(data['arr_0'])
                min20 = np.min(data['arr_0'])
                std20 = np.std(data['arr_0'])

            if os.path.isfile(file_path + "\\" + image_name[:-4]+file_extension[3]):
                data = np.load(file_path + "\\" + image_name[:-4]+file_extension[3])
                max40 = np.max(data['arr_0'])
                min40 = np.min(data['arr_0'])
                std40 = np.std(data['arr_0'])

            score_file.writerow({'image_name':image_name, 'class':image_class, 'score':prediction,
                                    '5_min':min5, '5_max':max5, '5_std':std5,
                                    '10_min':min10, '10_max':max10, '10_std':std10,
                                    '20_min':min20, '20_max':max20, '20_std':std20,
                                    '40_min':min40, '40_max':max40, '40_std':std40})
            print('done'+str(image_name))


#obtain_distribution(image_folder_path, image_name_list, image_class_list, extension)
#for i in range(0, len(extension)):
#    print(str(extension[i])+ str(result[i,:,0].mean())+ str(result[i,:,1].mean())+str(np.max(result[i,:,0]))+ str(np.min(result[i,:,1])) )



# region
def display_analysis(file_path, image_name_list, overlay,extension):



    file_extension=[]
    file_extension = extension.copy()


    for image_name in image_name_list:
        for i in range(0, len(extension)):
            file_extension[i] = image_name[:-4] + extension[i]

        rows = 5
        cols = 4
        axes=[]
        fig=plt.figure(figsize=(15 ,15))

        original_image = path_to_image_tensor((file_path + "\\" + image_name), 299, 299)
        axes.append(fig.add_subplot(rows, cols, 1))
        axes[-1].set_title(image_name)
        axes[-1].xaxis.set_visible(False)
        axes[-1].yaxis.set_visible(False)
        plt.imshow(original_image[0])


        for a in range(len(extension)):

            print(file_path + "\\" + file_extension[a])
            if os.path.isfile(file_path + "\\" + file_extension[a]):
                data = np.load(file_path + "\\" + file_extension[a])
                axes.append(fig.add_subplot(rows, cols, a + 2))
                axes[-1].set_title(extension[a][1:-9],y=1,pad=14)
                axes[-1].xaxis.set_visible(False)
                axes[-1].yaxis.set_visible(False)
                alpha_val = 1
                if overlay:
                    plt.imshow(original_image[0])
                    alpha_val = 0.5
                im=plt.imshow(data['arr_0'], cmap='jet', alpha=0.5)
                fig.colorbar(im,shrink=0.85)

            else:
                axes.append(fig.add_subplot(rows, cols, a + 2))
                axes[-1].set_title(extension[a][1:-9],y=1,pad=14 )
                axes[-1].xaxis.set_visible(False)
                axes[-1].yaxis.set_visible(False)
                im=plt.imshow(original_image[0])
                fig.colorbar(im,shrink=0.85)
                alpha_val=0.5

        #fig.tight_layout()
        fig.tight_layout()
        plt.show()

#display_analysis(image_folder_path, image_name_list, overlay, extension)

def display_specific_analysis(file_path, image_name_list, overlay,extension, title):
    file_extension=[]
    file_extension = extension.copy()

    rows=5
    cols=5
    axes = []
    #i = 0
    for i in range(len(extension)):
        fig = plt.figure(figsize=(9, 10))
        fig.suptitle(title+'_Neutrophil  '+extension[i][1:-9]+'_maps' , fontsize=16, y=1)

        for j in range(len(image_name_list)):

            file_extension = image_name_list[j][:-4] + extension[i]
            print(file_path + "\\" +file_extension)

            original_image = path_to_image_tensor((file_path + "\\" + image_name_list[j]), 299, 299)

            if os.path.isfile(file_path + "\\" + file_extension):
                data = np.load(file_path + "\\" + file_extension)

                new_data=data['arr_0']
                print(new_data.shape)
                new_data=cv2.resize(new_data,(262,262),interpolation=cv2.INTER_CUBIC)
                print(new_data.shape)
                axes.append(fig.add_subplot(rows, cols, j + 1))
                axes[-1].set_title(image_name_list[j][:-4], fontsize=8,y=1,pad=2)
                axes[-1].xaxis.set_visible(False)
                axes[-1].yaxis.set_visible(False)
                plt.imshow(original_image[0][18:-18,18:-18])
                im = plt.imshow(new_data, cmap='jet', alpha=0.5)
                #fig.colorbar(im, shrink=0.85)
        #fig.tight_layout()
        #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.subplots_adjust(left=0.001,right=0.999,top=0.96,bottom=0.001,wspace=0.005,hspace=0.05)
        plt.savefig(file_path + "\\" + title + extension[i][0:-4] + '.png')
        plt.show()

#display_specific_analysis(image_folder_path, image_name_list, overlay, extension, 'Normal')
#display_specific_analysis(image_folder_path, image_name_list_dysplastic, overlay, extension, 'Dysplastic')
#display_specific_analysis(image_folder_path, image_name_list_normal_m, overlay, extension, 'Normal_Misclassified')
#display_specific_analysis(image_folder_path, image_name_list_dysplastic_m, overlay, extension, 'Dysplastic_Misclassified')

def display_occlusion_analysis(file_path, image_name_list, overlay,extension, title):
    file_extension=[]
    file_extension = extension.copy()

    rows=6
    cols=4
    axes = []
    patch=['5','10','20','40']

    fig = plt.figure(figsize=(9, 13))
    for j in range(len(image_name_list)):

        fig.suptitle(title+'_Neutrophil  Occlusion_Sensitivity_maps' , fontsize=16, y=1)

        for i in range(len(extension)):

            file_extension = image_name_list[j][:-4] + extension[i]
            print(file_path + "\\" +file_extension)

            original_image = path_to_image_tensor((file_path + "\\" + image_name_list[j]), 299, 299)

            if os.path.isfile(file_path + "\\" + file_extension):
                data = np.load(file_path + "\\" + file_extension)

                new_data=data['arr_0']
                axes.append(fig.add_subplot(rows, cols, ((j*4)+(1+i))))
                if i==0:
                    axes[-1].set_title(image_name_list[j][:-4]+' ' +patch[i]+'px _patch', fontsize=8,y=1,pad=2)
                else:
                    axes[-1].set_title('patch' + patch[i], fontsize=8, y=1, pad=2)
                axes[-1].xaxis.set_visible(False)
                axes[-1].yaxis.set_visible(False)
                plt.imshow(original_image[0])
                im = plt.imshow(new_data, cmap='jet', alpha=0.5)
                cbar=fig.colorbar(im, shrink=0.99, format='%.6g', pad = 0.01)
                cbar.ax.tick_params(labelsize=6)
                #fig.tight_layout()
        #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.subplots_adjust(left=0.001,right=0.97,top=0.96,bottom=0.001,wspace=0.005,hspace=0.08)
    plt.savefig(file_path + "\\" + title + 'Occlusion_Sensitivity_maps.png')
    plt.show()

display_occlusion_analysis(image_folder_path, image_name_list_normal[0:6], overlay, extension, 'Normal_1')
display_occlusion_analysis(image_folder_path, image_name_list_dysplastic[0:6], overlay, extension, 'Dysplastic_1')
display_occlusion_analysis(image_folder_path, image_name_list_normal[7:13], overlay, extension, 'Normal_2')
display_occlusion_analysis(image_folder_path, image_name_list_dysplastic[7:13], overlay, extension, 'Dysplastic_2')
display_occlusion_analysis(image_folder_path, image_name_list_normal[14:20], overlay, extension, 'Normal_3')
display_occlusion_analysis(image_folder_path, image_name_list_dysplastic[14:20], overlay, extension, 'Dysplastic_3')
display_occlusion_analysis(image_folder_path, image_name_list_normal_m, overlay, extension, 'Normal_Misclassified')
display_occlusion_analysis(image_folder_path, image_name_list_dysplastic_m, overlay, extension, 'Dysplastic_Misclassified')
# endregion

def display_original_images(file_path, image_name_list, overlay,extension, title):
    file_extension=[]
    file_extension = extension.copy()

    rows=5
    cols=5
    axes = []
    #i = 0
    fig = plt.figure(figsize=(9, 10))
    fig.suptitle(title+'_Neutrophil' , fontsize=16, y=1)

    for j in range(len(image_name_list)):

        image_to_load = image_name_list[j]
        print(file_path + "\\" +image_to_load)
        original_image = path_to_image_tensor((file_path + "\\" + image_to_load), 299, 299)
        axes.append(fig.add_subplot(rows, cols, j + 1))
        axes[-1].set_title(image_to_load[:-4], fontsize=8,y=1,pad=2)
        axes[-1].xaxis.set_visible(False)
        axes[-1].yaxis.set_visible(False)
        plt.imshow(original_image[0][18:-18,18:-18])



    plt.subplots_adjust(left=0.001,right=0.999,top=0.96,bottom=0.001,wspace=0.005,hspace=0.05)
    plt.savefig(file_path + "\\" + title +'.png')
    plt.show()

#display_original_images(image_folder_path, image_name_list_normal, overlay, extension, 'Normal')
#display_original_images(image_folder_path, image_name_list_dysplastic, overlay, extension, 'Dysplastic')
#display_original_images(image_folder_path, image_name_list_normal_m, overlay, extension, 'Normal_Misclassified')
#display_original_images(image_folder_path, image_name_list_dysplastic_m, overlay, extension, 'Dysplastic_Misclassified')