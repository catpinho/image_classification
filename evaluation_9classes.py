import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import os

class_list = ["BOCAGEI", "CARBONELLI", "GUADARRAMAE", "HISPANICUS", "LIOLEPIS", "LUSITANICUS", "TUNESIACUS", "VAUCHERI", "VIRESCENS"]
labels = ["Pboc","Pcar","Pgua","Phis","Plio","Plus","Ptun","Pvau","Pvir"]
output_n = len(class_list)
# image size to scale down to (original images are 256 x 256 px)
img_width = 224
img_height = 224
target_size = [img_width, img_height]

valid_test_data_gen = ImageDataGenerator(rescale=1 / 255)

plt.rcParams.update({'font.size': 8})
#################################
dire = "/path_to_main_folder/"
res_file2 = open(dire + "all_res_best.txt", "a")
res_file2.write("model\tbest training accuracy\tbest training f1\tbest validation accuracy\tbest validation f1\tbest test accuracy\tbest test f1\n")


for mod in ["incv3","incrnv2","resnet50"]:
    res_file = open(dire + "FINAL_TABLE.txt", "a")

    if mod == "incv3":
        base_model = InceptionV3(weights = 'imagenet', include_top = False)
    elif mod == "resnet50":
        base_model = ResNet50(weights = 'imagenet', include_top = False)
    elif mod == "incrnv2":
        base_model = InceptionResNetV2(weights = 'imagenet', include_top = False)

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(units=output_n, activation='softmax')(x)


    model = keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
    preds=[]
    real=[]

    for fold in range(1,6):
        res_file2 = open(dire + "all_res_best.txt", "a")


# path to image folders, training with 60%, validation and test with 20% of images
        test_image_files_path = dire+ "cv" +str(fold)+ "_test/"
        train_image_files_path = dire+ "cv" +str(fold)+ "_train/"
        valid_image_files_path = dire+ "cv" +str(fold)+ "_valid/"

#Loading images
        test_image_array_gen = valid_test_data_gen.flow_from_directory (test_image_files_path,target_size = target_size, class_mode = "categorical", classes = class_list, shuffle = False)
        train_image_array_gen = valid_test_data_gen.flow_from_directory (train_image_files_path, target_size = target_size, class_mode = "categorical",
classes = class_list, shuffle = False)
        valid_image_array_gen = valid_test_data_gen.flow_from_directory (valid_image_files_path, target_size = target_size, class_mode = "categorical", classes = class_list, shuffle = False)
        
  r=[]
        rt=[]
        rv=[]
        for i in test_image_array_gen.classes:
            real.append(labels[i])
            r.append(labels[i])

        for iv in valid_image_array_gen.classes:
            rv.append(labels[iv])
        for it in train_image_array_gen.classes:
            rt.append(labels[it])
        weights = dire + "saved_models/" +mod+ "/best_" +mod+ "_big_cv"+str(fold)+"_D_M.h5"
        model.load_weights(weights)
        p=model.predict(test_image_array_gen)
       
        xp=[]
        for t in p:
            xp.append(labels[np.argmax(t)])
            preds.append(labels[np.argmax(t)])
        pt=model.predict(train_image_array_gen)
        xpt=[]
        for tt in pt:
            xpt.append(labels[np.argmax(tt)])
        pv=model.predict(valid_image_array_gen)
        xpv=[]
        for tv in pv:
            xpv.append(labels[np.argmax(tv)])

        tacc=model.evaluate(test_image_array_gen)[1]
        tracc=model.evaluate(train_image_array_gen)[1]
        vacc=model.evaluate(valid_image_array_gen)[1]
        f1=sklearn.metrics.f1_score(r,xp,average="weighted")
        all_f1=sklearn.metrics.f1_score(r,xp,average=None)
        tf1=sklearn.metrics.f1_score(rt,xpt,average="weighted")
        vf1=sklearn.metrics.f1_score(rv,xpv,average="weighted")

           res_file2.write(mod+"_cv"+str(fold)+"\t"+str(tracc)+"\t"+str(tf1)+"\t"+str(vacc)+"\t"+str(vf1)+"\t"+str(tacc)+"\t"+str(f1))
        for s in all_f1:
            res_file2.write("\t"+str(s))
        res_file2.write("\n")
        res_file2.close()

    
    total_acc=sklearn.metrics.accuracy_score(real,preds)
    total_f1m=sklearn.metrics.f1_score(real,preds,average="macro")
    total_f1=sklearn.metrics.f1_score(real,preds,average="weighted")
    per_species_f1=sklearn.metrics.f1_score(real,preds,average=None)
    #non normalized confusion matrix
    cm=sklearn.metrics.confusion_matrix(real,preds,labels=labels)
    #normalized confusion matrix over rows
ncm=sklearn.metrics.confusion_matrix(real, preds, labels=labels, normalize="true")
    disp=sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot()
plt.savefig(dire+mod+"_best.png")
    dispn=sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=ncm,display_labels=labels)
dispn.plot()
plt.savefig(dire+mod+"_n_best.png")
    res_file.write(mod+"\t"+str(total_acc)+"\t"+str(total_f1m)+"\t"+str(total_f1))
for i in per_species_f1:
       res_file.write("\t"+str(i))
res_file.write("\n")
res_file.close()