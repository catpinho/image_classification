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
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os

class_list = ["BOCAGEI" , "LUSITANICUS"]

# image size to scale down to (original images are 256 x 256 px)
img_width = 224
img_height = 224
target_size = [img_width, img_height]

valid_test_data_gen = ImageDataGenerator(rescale=1 / 255)

#path to where the images are
dire=”/yourpath/” 

res_file2 = open(dire + "all_res.txt", "a")
res_file2.write("model and cv\ttraining accuracy\ttraining auc\ttraining f1\tvalidation accuracy\tvalidation auc\tvalidation f1\ttest accuracy\ttest auc\ttest f1\n")


for mod in ["incv3","incrnv2","resnet50"]:
	

    #file with global results across cross-validation sets
res_file = open(dire + "FINAL_TABLE.txt", "a")


        ### pre-trained model
    ### create the base pre-trained model
    if mod == "incv3":
        base_model = InceptionV3(weights = 'imagenet', include_top = False)
    elif mod == "resnet50":
        base_model = ResNet50(weights = 'imagenet', include_top = False)
    elif mod == "incrnv2":
        base_model = InceptionResNetV2(weights = 'imagenet', include_top = False)

    x = base_model.output
    ### add a global spatial average pooling layer
    x = keras.layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = keras.layers.Dense(1024, activation='relu')(x)
    # add dropout
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(units = 1, activation='sigmoid')(x)

    model = keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001,decay=1e-6), loss="binary_crossentropy", metrics=['binary_accuracy'])

    probs=[]
    real=[]

    for fold in range(1,6):
	#file with all the information for each cross-validation set
        res_file2 = open(dire + "all_res.txt", "a")


# path to image folders, training with 80% and test with 20% of images
        test_image_files_path = dire+ "cv" +str(fold)+ "_test/"
        train_image_files_path = dire+ "cv" +str(fold)+ "_train/"
        valid_image_files_path = dire+ "cv" +str(fold)+ "_valid/"

#Loading images
        test_image_array_gen = valid_test_data_gen.flow_from_directory (test_image_files_path, target_size = target_size, class_mode = "binary", classes = class_list, shuffle = False)
        train_image_array_gen = valid_test_data_gen.flow_from_directory (train_image_files_path, target_size = target_size, class_mode = "binary", classes = class_list, shuffle = False)
        valid_image_array_gen = valid_test_data_gen.flow_from_directory (valid_image_files_path, target_size = target_size, class_mode = "binary", classes = class_list, shuffle = False)

#collect real labels from the three sets:
        r=[]
        rt=[]
        rv=[]
        for i in test_image_array_gen.classes:
            real.append(i)
            r.append(i)
        for iv in valid_image_array_gen.classes:
            rv.append(iv)
        for it in train_image_array_gen.classes:
            rt.append(it)

#now collect models saved an the end of the learning process
weights = dire + "saved_models/" + mod + "/final_" + mod + "_ cv"+str(fold)+"_H_M.h5"
        model.load_weights(weights)
        p=model.predict(test_image_array_gen)
        xp=[]
        for t in p:
            xp.append(int(round(t[0],0)))
        pt=model.predict(train_image_array_gen)
        xpt=[]
        for tt in pt:
            xpt.append(int(round(tt[0],0)))
        pv=model.predict(valid_image_array_gen)
        xpv=[]
        for tv in pv:
            xpv.append(int(round(tv[0],0)))
        pr=[]
        prt=[]
        prv=[]
        for m in p:
            probs.append(m[0])
            pr.append(m[0])
        for mt in pt:
            prt.append(mt[0])
        for mv in pv:
            prv.append(mv[0])

        tacc=model.evaluate(test_image_array_gen)[1]
        tracc=model.evaluate(train_image_array_gen)[1]
        vacc=model.evaluate(valid_image_array_gen)[1]
        auc=sklearn.metrics.roc_auc_score(r,pr)
        f1=sklearn.metrics.f1_score(r,xp)
        tauc=sklearn.metrics.roc_auc_score(rt,prt)
        tf1=sklearn.metrics.f1_score(rt,xpt)
        vauc=sklearn.metrics.roc_auc_score(rv,prv)
        vf1=sklearn.metrics.f1_score(rv,xpv)

           res_file2.write(mod+"_cv"+str(fold)+"\t"+str(tracc)+"\t"+str(tauc)+"\t"+str(tf1)+"\t"+str(vacc)+"\t"+str(vauc)+"\t"+str(vf1)+"\t"+str(tacc)+"\t"+str(auc)+"\t"+str(f1)+"\n")
        res_file2.close()
    
    
preds=[]
    for pr in probs:
        preds.append(int(round(pr)))
    
total_acc=sklearn.metrics.accuracy_score(real,preds)
total_auc=sklearn.metrics.roc_auc_score(real,probs)
total_f1=sklearn.metrics.f1_score(real,preds,pos_label=1)
total_f10=sklearn.metrics.f1_score(real,preds,pos_label=0)
    res_file.write(mod+"\t"+str(total_acc)+"\t"+str(total_auc)+"\t"+str(total_f1)+"\t"+str(total_f10)+"\n")
res_file.close()
