
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import cv2

file_path=("C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\Annotation")
image_name_vect=["TD_BNE_2249981.jpg",
                 #"TD_BNE_2256328.jpg",
                 #"TD_SNE_741897.jpg",
                 #"TN_SNE_14872661.jpg",
                 #"TN_SNE_118039.jpg",
                 #"TN_SNE_14872673.jpg"
                ]
    #load file

for image_name in image_name_vect :

    json_mask_data = os.path.join(file_path+"\\"+image_name[:-4]+"_json.txt")
    if os.path.isfile(json_mask_data):  # file with notation shapes exists
        with open(json_mask_data) as json_file:
            masks_data = json.load(json_file)
        image_saved_data = masks_data

        original_image_x=360
        original_image_y=363

        #initialisation
        fig, ax1 = plt.subplots(figsize=(10,10))
        fig2, ax2 = plt.subplots(figsize=(10,10))
        ax1.set_xlim(0, original_image_x)
        ax1.set_ylim(0, original_image_y)
        fig.gca().invert_yaxis()
        ax2.set_xlim(0, original_image_x)
        ax2.set_ylim(0, original_image_y)
        fig2.gca().invert_yaxis()

        for i in range(0,len(image_saved_data["shapes"])):

            plotly_path=image_saved_data["shapes"][i]["path"]
            plotly_color=image_saved_data["shapes"][i]["line"]["color"]
            plotly_width=image_saved_data["shapes"][i]["line"]["width"]
            print(plotly_color)


            codes=[]
            verts=[]

            path_string=plotly_path.split("L")
            for i in range(0,len(path_string)):
                print(path_string[i])
                position = []
                if "M" in path_string[i]:
                    path_string[i]=path_string[i].replace('M', '')
                    x_coor,y_coor= path_string[i].split(",")
                    x_coor=float(x_coor)
                    y_coor = float(y_coor)
                    position.insert(0,x_coor)
                    position.insert(1,y_coor)
                    verts.insert(i,tuple(position))
                    codes.insert(i,np.uint8(1))

                elif "Z" in path_string[i]:
                    print(path_string[i])
                    path_string[i]=path_string[i].replace('Z', '')
                    x_coor, y_coor = path_string[i].split(",")
                    x_coor = float(x_coor)
                    y_coor = float(y_coor)
                    position.insert(0, x_coor)
                    position.insert(1, y_coor)
                    verts.insert(i, tuple(position))
                    codes.insert(i,np.uint8(79))

                else:
                    #"L"
                    x_coor, y_coor = path_string[i].split(",")
                    x_coor = float(x_coor)
                    y_coor = float(y_coor)
                    position.insert(0, (x_coor))
                    position.insert(1, (y_coor))
                    verts.insert(i, tuple(position))
                    codes.insert(i,np.uint8(2))

            path = Path(verts, codes)

            patch = patches.PathPatch(path,edgecolor=plotly_color, facecolor=plotly_color, lw=plotly_width)
            if plotly_color=='#cb4335':
                ax1.add_patch(patch)
            if plotly_color=='#5dade2':
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

        fig.savefig(file_path + '\\' + image_name[:-4] + '_annotation_cytoplasm_mask.png', bbox_inches='tight', format='png')
        fig2.savefig(file_path + '\\' + image_name[:-4] + '_annotation_nucleus_mask.png', bbox_inches='tight', format='png')

        im = cv2.imread(file_path + '\\' + image_name[:-4] + '_annotation_nucleus_mask.png')
        im = cv2.resize(im, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        print (type(im), im.shape,np.unique(im, return_counts=True), np.unique(im))
        np.savez((file_path + '\\' + image_name[:-4] + '_annotation_nucleus_mask.npz'), im)

        im = cv2.imread(file_path + '\\' + image_name[:-4] + '_annotation_cytoplasm_mask.png')
        im = cv2.resize(im, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        print(type(im), im.shape, np.unique(im, return_counts=True), np.unique(im))
        np.savez((file_path + '\\' + image_name[:-4] + '_annotation_cytoplasm_mask.npz'), im)

        #print plot
        plt.show()
        plt.clf()






