#required modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os

class_list = ["BOCAGEI", "CARBONELLI", "GUADARRAMAE", "HISPANICUS", "LIOLEPIS", "LUSITANICUS","TUNESIACUS", "VAUCHERI", "VIRESCENS"]

# number of output classes 
output_n = len(class_list)

# image size to scale down to (original images were 256 x 256 px)
img_width = 224
img_height = 224
target_size = [img_width, img_height]

# RGB = 3 channels
channels = 3

# path to where the images are stored, already in different folders for each cross-validation split; need to adjust for windows-style paths
dire = "/your_path/"

# intermediate result file used for monitoring
res_file = open(dire+"summary_results_M_DORSAL.txt","w")
res_file.write("data_set\tbest_train_acc\tfinal_train_acc\tbest_val_acc\tfinal_val_acc\tbest_test_acc\tfinal_test_acc\n")
res_file.close()

mods=["incv3","resnet50","incrnv2"]:
    
#for each of the models
for mod in mods:
    
    #for each cross-validation turn:
    for fold in range(1,6):

        res_file = open(dire+"summary_results_M_DORSAL.txt","a")

    # path to image folders, training with 60% and valid and test with 20% of images
        train_image_files_path = dire+"cv"+str(fold)+"_train/"
        valid_image_files_path = dire+"cv"+str(fold)+"_valid/"
        test_image_files_path = dire+"cv"+str(fold)+ "_test/"

    #Loading images

    # data augmentation in training set
        train_data_gen = ImageDataGenerator(
          rescale = 1/255,
          rotation_range = 70,
          width_shift_range = 0.5,
          height_shift_range = 0.5,
          shear_range = 0.5,
          zoom_range = 0.5,
          brightness_range = [0.2,1.8])
        
    # validation data is not augmented but only rescaled. The same protocol is used for test images as well
        valid_test_data_gen = ImageDataGenerator(
            rescale = 1/255)

    #loading training images from its directory:
        train_image_array_gen = train_data_gen.flow_from_directory (train_image_files_path, target_size = target_size,class_mode = "categorical", classes = class_list)

    #loading validation images
        valid_image_array_gen = valid_test_data_gen.flow_from_directory (valid_image_files_path, target_size = target_size,class_mode = "categorical",classes = class_list)

    # loading test images. The shuffle=False is so that the prediction matches the same order they are in the directories and we can compare predictions with real classes (otherwise it is random)
        test_image_array_gen = valid_test_data_gen.flow_from_directory (test_image_files_path, target_size = target_size, class_mode = "categorical",classes = class_list, shuffle = False)
        
        ### number of training samples
        train_samples = train_image_array_gen.n
        ### number of validation samples
        valid_samples = valid_image_array_gen.n
        
        
        batch_size=64
        
    #to account for unbalanced classes
        class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(train_image_array_gen.classes), train_image_array_gen.classes)
        class_weights = dict(enumerate(class_weight))

    #path where to save the models along the run
        checkpoint_path = dire+ "saved_models/" + mod + "/best_"+mod+"_cv" + str(fold) + "_D_M.h5"

        # Create checkpoint callback (we only save the best model, replacing it as it improves validation scores)
        ckp=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        )

        # Create a log folder for visualizaing the run using tensorboard

        tsb=tf.keras.callbacks.TensorBoard(
            log_dir=dire+"logs/"+mod+"/logs_"+mod+"_cv"+str(fold),
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        )

        ###TRAINING
        

        ### create the base pre-trained model
        if mod == "incv3":
            base_model = InceptionV3(weights = 'imagenet', include_top = False)
        elif mod == "resnet50":
            base_model = ResNet50(weights = 'imagenet', include_top = False)
        elif mod == "incrnv2":
            base_model = InceptionResNetV2(weights = 'imagenet', include_top = False)

        ### modifications to the base model were the same in all cases:

        #add a global spatial average pooling layer
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        # adding a fully-connected layer
        x = keras.layers.Dense(1024, activation='relu')(x)
        # add dropout
        x = keras.layers.Dropout(0.5)(x)
        predictions = keras.layers.Dense(units = output_n, activation='softmax')(x)

        # this is the model we will train
        model = keras.Model(inputs=base_model.input, outputs=predictions)

        #all layers in the model are trainable

        for layer in model.layers:
            layer.trainable=True
            
        ### we need to recompile the model before using it
        ### we use adam with a low learning rate
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])

        ### fitting the model with 2000 epochs.
        model.fit(
            train_image_array_gen,
            steps_per_epoch=train_samples // batch_size,
            epochs=2000,
            validation_data=valid_image_array_gen,
            validation_steps=valid_samples // batch_size,
            class_weight=class_weights,
            callbacks=[ckp,tsb],
            verbose=2)
        
        ##save the final model after the 2000 epochs (does not replace the "best" model)
        model.save_weights(dire+"saved_models/"+mod+"/final_"+mod+"_cv"+str(fold)+"_D_M.h5")

        #write some statistics for monitoring

        #these are the statistics based on the final model
        train_f=model.evaluate(train_image_array_gen)[1]
        val_f=model.evaluate(valid_image_array_gen)[1]
        tes_f=model.evaluate(test_image_array_gen)[1]
        #and these are the statistics after loading the best model
        model.load_weights(checkpoint_path)
        train_b=model.evaluate(train_image_array_gen)[1]
        val_b=model.evaluate(valid_image_array_gen)[1]
        tes_b=model.evaluate(test_image_array_gen)[1]

    #write these to the monitoring file
        res_file.write(mod+"_cv"+str(fold)+"\t"+str(train_b)+"\t"+str(train_f)+"\t"+str(val_b)+"\t"+str(val_f)+"\t"+str(tes_b)+"\t"+str(tes_f)+"\n")
       res_file.close()