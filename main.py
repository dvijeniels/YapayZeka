import sys,os,glob, cv2, skimage
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from ui_AraYuz import *
from PyQt5.QtGui import QPixmap
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import tensorflow as tf
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout , MaxPool2D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report,roc_auc_score, roc_curve,confusion_matrix,accuracy_score,auc
from keras.utils import img_to_array, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from keras.utils.np_utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle as shf
from tqdm import tqdm

class window(QtWidgets.QMainWindow):
    def __init__(self):
        self.train_dir=""
        self.val_dir=""
        self.test_dir=""
        self.test_data=[]
        self.test_labels = []
        self.file=[0]
        self.h5file=[0]
        self.fname=[0]
        super(window,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.modelEgit.clicked.connect(self.modelEgitiliyor)
        self.ui.datasetGoster.clicked.connect(self.AddVeriToListView)
        self.ui.btnHoldout.clicked.connect(self.HoldOut)
        self.ui.testingImage.clicked.connect(self.TestingImage)
        self.ui.siniflandir.clicked.connect(self.Siniflandirma)
        self.ui.btnConfusionMatrix.clicked.connect(self.Matrix)
        self.ui.btnVeriCoklama.clicked.connect(self.VeriCoklama)
        self.ui.btnKFold.clicked.connect(self.KFold)
        self.ui.btnRoc.clicked.connect(self.RocEgrisi)
        self.ui.btnMetrics.clicked.connect(self.Metrics)
        self.ui.btnKayipMatris.clicked.connect(self.KayipMatrisi)
        self.ui.btnDogruluk.clicked.connect(self.DogrulukMatrisi)
        self.ui.menu_New = QMenu()
        self.ui.action_Resim_yukle.triggered.connect(self.resimAc)
        self.ui.action_Open_file.triggered.connect(self.h5Ac)
        self.ui.action_Open.triggered.connect(self.dosyaAc)
        
    def HoldOut(self):
        imageSize=244
        X = []
        y = []
        for folderName in os.listdir(self.train_dir):
            if not folderName.startswith('.'):
                if folderName in ['NORMAL']:
                    label = 0
                elif folderName in ['PNEUMONIA']:
                    label = 1
                else:
                    label = 2
                for image_filename in tqdm(os.listdir(self.train_dir + folderName)):
                    img_file = cv2.imread(self.train_dir + folderName + '/' + image_filename)
                    if img_file is not None:
                        img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                        img_arr = np.asarray(img_file)
                        X.append(img_arr)
                        y.append(label)
        X = np.asarray(X)
        y = np.asarray(y)
        def model_create():
            cnnHoldOut = Sequential([
            Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[244, 244, 3]),
            MaxPool2D(pool_size=2, strides=2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPool2D(pool_size=2, strides=2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=1, activation='sigmoid')])
            cnnHoldOut.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            return cnnHoldOut

        holdout = 10
        cv_acc= []
        cv_loss= []

        # X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2, random_state=100)
        for train_index,test_index in train_test_split(X,y,test_size=0.2, random_state=100):
            X_train,X_test=X[train_index],X[test_index]
            Y_train,Y_test=y[train_index],y[test_index]
            
            models=model_create()
            
            print(f' Fold_no {holdout}')

            history_cnn = models.fit(
                        X_train, Y_train, epochs = 10, batch_size = 32, validation_data=(X_test,Y_test))
            #scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            
            models.save("C:\\Users\\djadi\\chest_xray\\Result\\CNN_holdout" +str(holdout)+ ".h5")
            scores = models.evaluate(X_test, Y_test, verbose=0)
            cv_acc.append(scores[1] * 100)
            cv_loss.append(scores[0])
            holdout = holdout + 1
        print('Sonuçlar')
        for i in range(0, len(cv_acc)):
            self.ui.listViewVeriler.addItem(f'Holdout {i+1} - Loss: {cv_loss[i]} - Accuracy: %{cv_acc[i]}')
        self.ui.listViewVeriler.addItem("Ortalama sonuçlar:")
        self.ui.listViewVeriler.addItem(f'Accuracy: {np.mean(cv_acc)} (+- {np.std(cv_acc)})')
        self.ui.listViewVeriler.addItem(f' Loss: {np.mean(cv_loss)}')
        
    def VeriCoklama(self):
        def isCheck_Klasor(path):
            if not os.path.isdir(path):
                os.mkdir(path)
        classes=os.listdir(self.val_dir)
        datagen = ImageDataGenerator(rotation_range=180,width_shift_range=0.3,height_shift_range=0.3,
        shear_range=0.15,zoom_range=0.5,horizontal_flip=True)
        datagen2 = ImageDataGenerator()
        path1='C:/Users/djadi/chest_xray/Result/VerilerCoklanmisHali/'
        isCheck_Klasor(path1)

        for class_name in classes:
            
            
            
            path2=self.val_dir+class_name+'/'
            files=os.listdir(path2)
            
            if self.ui.veriRotation.isChecked()==True:
                veriRotation='C:/Users/djadi/chest_xray/Result/VerilerCoklanmisHali/veriRotation/'
                isCheck_Klasor(veriRotation)
                save_here = veriRotation+class_name
                isCheck_Klasor(save_here)
                
                for file in files:
                    image_path = path2+file

                    image = cv2.imread(image_path)
                    image = np.expand_dims(image, 0) 
                    datagen.fit(image)
                    
                    if class_name=="NORMAL":
                        range_value=3
                    elif class_name=="PNEUMONIA":
                        range_value=9

                    for x, val in zip(datagen.flow(image,save_to_dir=save_here,save_prefix='aug',save_format='jpeg'),range(range_value)):
                        pass
                
                    print (class_name,file, " islem bitti...")
            if self.ui.veriGriGauss.isChecked()==True:
                veriGriGauss='C:/Users/djadi/chest_xray/Result/VerilerCoklanmisHali/veriGriGauss/'
                isCheck_Klasor(veriGriGauss)
                save_here = veriGriGauss+class_name
                isCheck_Klasor(save_here)
                for file in files:
                    image_path = path2+file

                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    image = cv2.addWeighted (image, 4, cv2.GaussianBlur(image, (0,0), 512/10), -4, 128)
                    image = np.expand_dims(image, 0) 
                    datagen2.fit(image)
                    
                    if class_name=="NORMAL":
                        range_value=3
                    elif class_name=="PNEUMONIA":
                        range_value=9

                    for x, val in zip(datagen2.flow(image,save_to_dir=save_here,save_prefix='aug',save_format='jpeg'),range(range_value)):
                        pass
                
                    print (class_name,file, " islem bitti...")
            if self.ui.veriGenisletilmesi.isChecked()==True:
                veriGenisletilmesi='C:/Users/djadi/chest_xray/Result/VerilerCoklanmisHali/veriGenisletilmesi/'
                isCheck_Klasor(veriGenisletilmesi)
                save_here = veriGenisletilmesi+class_name
                isCheck_Klasor(save_here)
                for file in files:
                    image_path = path2+file

                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    kernel = np.ones((5, 5), np.uint8)
                    img_erosion = cv2.dilate(image, kernel, iterations=3)
                    image = np.expand_dims(img_erosion, 0) 
                    datagen2.fit(image)
                    
                    if class_name=="NORMAL":
                        range_value=3
                    elif class_name=="PNEUMONIA":
                        range_value=9

                    for x, val in zip(datagen2.flow(image,save_to_dir=save_here,save_prefix='aug',save_format='jpeg'),range(range_value)):
                        pass
                
                    print (class_name,file, " islem bitti...")
            if self.ui.veriErosion.isChecked()==True:
                veriErosion='C:/Users/djadi/chest_xray/Result/VerilerCoklanmisHali/veriErosion/'
                isCheck_Klasor(veriErosion)
                save_here = veriErosion+class_name
                isCheck_Klasor(save_here)
                for file in files:
                    image_path = path2+file

                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    kernel = np.ones((5, 5), np.uint8)
                    img_erosion = cv2.erode(image, kernel, iterations=3)
                    image = np.expand_dims(img_erosion, 0) 
                    datagen2.fit(image)
                    
                    if class_name=="NORMAL":
                        range_value=3
                    elif class_name=="PNEUMONIA":
                        range_value=9

                    for x, val in zip(datagen2.flow(image,save_to_dir=save_here,save_prefix='aug',save_format='jpeg'),range(range_value)):
                        pass
                
                    print (class_name,file, " islem bitti...")
        
    def SelectKFold(self):
        if self.ui.comboBoxKFold.currentIndex()==1:
            return 5
        elif self.ui.comboBoxKFold.currentIndex()==2:
            return 10
        elif self.ui.comboBoxKFold.currentIndex()==3:
            return 15
    
    def KFold(self):
        if self.file[0] != 0:
            fold_no=self.SelectKFold()
            if fold_no!=None:
                imageSize=244
                X = []
                y = []
                for folderName in os.listdir(self.test_dir):
                    if not folderName.startswith('.'):
                        if folderName in ['NORMAL']:
                            label = 0
                        elif folderName in ['PNEUMONIA']:
                            label = 1
                        else:
                            label = 2
                        for image_filename in tqdm(os.listdir(self.test_dir + folderName)):
                            img_file = cv2.imread(self.test_dir + folderName + '/' + image_filename)
                            if img_file is not None:
                                img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                                img_arr = np.asarray(img_file)
                                X.append(img_arr)
                                y.append(label)
                X = np.asarray(X)
                y = np.asarray(y)
                def model_create():
                    cnn = Sequential([
                    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[244, 244, 3]),
                    MaxPool2D(pool_size=2, strides=2),
                    Conv2D(filters=32, kernel_size=3, activation='relu'),
                    MaxPool2D(pool_size=2, strides=2),
                    Flatten(),
                    Dense(units=128, activation='relu'),
                    Dense(units=1, activation='sigmoid')])
                    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                    return cnn


                
                strkf=StratifiedKFold(n_splits=fold_no, random_state = 11, shuffle = True)

                X = np.asarray(X)
                y = np.asarray(y)

                fold = 1
                cv_acc= []
                cv_loss= []

                for train_index,test_index in strkf.split(X,y):
                    X_train,X_test=X[train_index],X[test_index]
                    Y_train,Y_test=y[train_index],y[test_index]

                    models=model_create()
                    
                    print(f' Fold_no {fold}')

                    history_cnn = models.fit(
                                X_train, Y_train, epochs = int(self.ui.txtEpoch.text()), batch_size = int(self.ui.txtBatchSize.text()), validation_data=(X_test,Y_test))
                    
                    models.save("C:\\Users\djadi\\chest_xray\\Result\\K_Fold_Sonuclar\\CNN_kfold" +str(fold)+ ".h5")
                    scores = models.evaluate(X_test, Y_test, verbose=0)
                    cv_acc.append(scores[1] * 100)
                    cv_loss.append(scores[0])
                    fold = fold + 1
                self.ui.listViewVeriler.addItem("Sonuçlar")
                for i in range(0, len(cv_acc)):
                    self.ui.listViewVeriler.addItem(f' Fold {i+1} - Loss: {cv_loss[i]} - Accuracy: %{cv_acc[i]}')
                self.ui.listViewVeriler.addItem("Ortalama sonuçlar:")
                self.ui.listViewVeriler.addItem(f'Accuracy: {np.mean(cv_acc)} (+- {np.std(cv_acc)})')
                self.ui.listViewVeriler.addItem(f' Loss: {np.mean(cv_loss)}')
            else:
                msg=QMessageBox()
                msg.setWindowTitle("Uyarı")
                msg.setText("KFold seçiniz")
                x=msg.exec_()
        else:
            self.show_popup()
    
    
    def plot_cm(predictions, y_test, title):
        labels = ['Normal', 'Pnuemonia']
        cm = confusion_matrix(y_test,predictions)
        cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
        plt.figure(figsize = (7,7))
        plt.title(title)
        sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)
        plt.show()
    
    def dosyaAc(self):
        self.file = str(QFileDialog.getExistingDirectory(self, "Dosya seç"))
        self.ui.listViewVeriler.addItem("Seçilen dosya adı :"+self.file)
        self.train_dir=self.file+"/train/"
        self.val_dir=self.file+"/val/"
        self.test_dir=self.file+"/test/"
        self.image_siniflandirma()
    
    def h5Ac(self):
        self.h5file,_=QFileDialog.getOpenFileName(self,"h5 seç","", "h5 (*.h5)")
        self.ui.listViewVeriler.addItem("Seçilen h5 :"+self.h5file)
    
    def show_popup(self):
        msg=QMessageBox()
        if self.file[0]==0:
            msg.setWindowTitle("Uyarı")
            msg.setText("Dosya seçim  yapılmadı!")
        elif self.h5file[0] == 0:
            msg.setWindowTitle("Uyarı")
            msg.setText("h5 file seçim  yapılmadı!")
        elif self.fname[0] == 0:
            msg.setWindowTitle("Uyarı")
            msg.setText("Resim seçim  yapılmadı!")
        else: 
            msg.setWindowTitle("Uyarı")
            msg.setText("Hatalı işlemdir")
        x=msg.exec_()
        
    def resimAc(self):
        self.fname=QFileDialog.getOpenFileName(self,"Resim seç","", "RESIMLER (*.png;*.jpg;*.jpeg)")
        self.pixmap=QPixmap(self.fname[0])
        scaled = self.pixmap.scaled(self.ui.SelectedImage.size(), QtCore.Qt.KeepAspectRatio)
        self.ui.SelectedImage.setPixmap(scaled)
        sp = self.ui.SelectedImage.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Maximum)
        self.ui.SelectedImage.setSizePolicy(sp)
        self.layout().setAlignment(self.ui.SelectedImage, QtCore.Qt.AlignCenter)
        
        
        
    def Metrics(self):
        precision = self.tp / (self.tp + self.fp) * 100
        recall = self.tp / (self.tp + self.fn) * 100
        self.ui.listViewVeriler.addItems(["〰"*30])
        self.ui.listViewVeriler.addItems(["Accuracy: {}%".format(self.acc)])
        self.ui.listViewVeriler.addItems(["Precision: {}%".format(precision)])
        self.ui.listViewVeriler.addItems(["Recall: {}%".format(recall)])
        self.ui.listViewVeriler.addItems(["F1-score: {}".format(2 * precision * recall / (precision + recall))])
        self.ui.listViewVeriler.addItems(["〰"*30])
       
    def Siniflandirma(self):
        if self.file[0] != 0:
            X_train = []
            y_train = []
            code = {'NORMAL':0 ,'PNEUMONIA':1}
            for folder in  os.listdir(self.train_dir) : 
                #files = glob.glob(str(self.train_dir + folder + '/*.jpeg'))
                excel_path = [os.path.normpath(i) for i in glob.glob(str(self.train_dir + folder + '/*'))]
                #print(files)
                for file in excel_path: 
                    new_string = file.replace("\\", "/")
                    image = cv2.imread(new_string)
                    image_array = cv2.resize(image , (244,244))
                    X_train.append(list(image_array))
                    y_train.append(code[folder])
            np.save('X_train',X_train)
            np.save('y_train',y_train)

            X_test = []
            y_test = []
            for folder in  os.listdir(self.test_dir) : 
                excel_path = [os.path.normpath(i) for i in glob.glob(str(self.test_dir + folder + '/*'))]
                #files = glob.glob(str(self.test_dir + folder + '/*'))
                for file in excel_path: 
                    image = cv2.imread(file)
                    image_array = cv2.resize(image , (244,244))
                    X_test.append(list(image_array))
                    y_test.append(code[folder])
            np.save('X_test',X_test)
            np.save('y_test',y_test)
            loaded_X_train = np.load('./X_train.npy')
            loaded_X_test = np.load('./X_test.npy')
            loaded_y_train = np.load('./y_train.npy')
            loaded_y_test = np.load('./y_test.npy')

            X_train = loaded_X_train.reshape([-1, np.product((244,244,3))])
            X_test = loaded_X_test.reshape([-1, np.product((244,244,3))])
            y_train = loaded_y_train
            y_test = loaded_y_test
            X_train, y_train = shf(X_train, y_train, random_state=15)
            X_test, y_test = shf(X_test, y_test, random_state=15)
            
            if self.ui.KNeighborsClassifier.isChecked()==True:
                knn = KNeighborsClassifier(n_neighbors=10)
                knn.fit(X_train, y_train)
                knn_predcited = knn.predict(X_test)
                self.ui.listViewVeriler.addItem("KNN accuracy score is: " + str(knn.score(X_test, y_test)))
                #self.plot_cm(knn_predcited, y_test, "KNN Confusion Matrix")
                labels = ['Normal', 'Pnuemonia']
                cm = confusion_matrix(y_test,knn_predcited)
                cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
                plt.figure(figsize = (7,7))
                plt.title("KNN Confusion Matrix")
                sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)
                plt.show()
                
            if self.ui.LogisticRegression.isChecked()==True:
                log_reg  = LogisticRegression(solver='lbfgs', max_iter=100)
                log_reg.fit(X_train, y_train)
                log_reg_predcited = log_reg.predict(X_test)
                self.ui.listViewVeriler.addItem('Logistic Regression accuracy score is: ' + str(log_reg.score(X_test, y_test)))
                #self.plot_cm(log_reg_predcited, y_test, 'Logistic Regression Confusion Matrix') 
                labels = ['Normal', 'Pnuemonia']
                cm = confusion_matrix(y_test,log_reg_predcited)
                cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
                plt.figure(figsize = (7,7))
                plt.title("Logistic Regression Confusion Matrix")
                sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)
                plt.show()
                
            if self.ui.DecisionTreeClassifier.isChecked()==True:
                dtc  = DecisionTreeClassifier()
                dtc.fit(X_train, y_train)
                dtc_predcited = dtc.predict(X_test)
                self.ui.listViewVeriler.addItem('Decision Tree Classifier accuracy score is: ' + str(dtc.score(X_test, y_test)))
                #self.plot_cm(dtc_predcited, y_test, 'Decision Tree Confusion Matrix')
                labels = ['Normal', 'Pnuemonia']
                cm = confusion_matrix(y_test,dtc_predcited)
                cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
                plt.figure(figsize = (7,7))
                plt.title("Decision Tree Confusion Matrix")
                sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)
                plt.show()
        
            if self.ui.RandomForestClassifier.isChecked()==True:
                rfc = RandomForestClassifier(max_depth=2, random_state=0)
                rfc.fit(X_train, y_train)
                rfc_predcited = rfc.predict(X_test)
                self.ui.listViewVeriler.addItem('Random forests Classifier accuracy score is: ' + str(rfc.score(X_test, y_test)))
                #self.plot_cm(rfc_predcited, y_test, 'Random Forests Confusion Matrix')
                labels = ['Normal', 'Pnuemonia']
                cm = confusion_matrix(y_test,rfc_predcited)
                cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
                plt.figure(figsize = (7,7))
                plt.title("Random Forests Confusion Matrix")
                sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)
                plt.show()
        
            if self.ui.SVM.isChecked()==True:
                svm = SVC(max_iter=100)
                svm.fit(X_train, y_train)
                svm_predcited = svm.predict(X_test)
                self.ui.listViewVeriler.addItem('Support Vector Machine Classifier accuracy score is: ' + str(svm.score(X_test, y_test)))
                #self.plot_cm(svm_predcited, y_test, 'Support Vector Machine Confusion Matrix')
                labels = ['Normal', 'Pnuemonia']
                cm = confusion_matrix(y_test,svm_predcited)
                cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
                plt.figure(figsize = (7,7))
                plt.title("Support Vector Machine Confusion Matrix")
                sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)
                plt.show()
            else: 
                msg=QMessageBox()
                msg.setWindowTitle("Uyarı")
                msg.setText("Seçim  yapılmadı!")
        else: self.show_popup()
    
    def RocEgrisi(self):
        if self.h5file[0]!=0:
            historyCNN=tf.keras.models.load_model(self.h5file)
            predictions = historyCNN.predict(self.test_data)
            fpr, tpr, threshold = metrics.roc_curve(self.test_labels, predictions)
            roc_auc = metrics.auc(fpr, tpr)
            plt.title('Modelimiz için Roc eğrisi')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
        else: self.show_popup()
         
    def modelEgitiliyor(self):
        if self.file[0]!=0:
            train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
            training_set = train_datagen.flow_from_directory(self.train_dir,
                                                            target_size = (244, 244),
                                                            batch_size = int(self.ui.txtBatchSize.text()),
                                                            class_mode = 'binary')
            test_datagen = ImageDataGenerator(rescale = 1./255)
            test_set = test_datagen.flow_from_directory(self.test_dir,
                                                        target_size = (244, 244),
                                                        batch_size = int(self.ui.txtBatchSize.text()),
                                                        class_mode = 'binary')
            self.cnn = tf.keras.models.Sequential()
            self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[244, 244, 3]))
            self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
            self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            self.cnn.add(tf.keras.layers.Flatten())
            self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
            self.cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
            self.cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            
            self.historyCNN=self.cnn.fit(x = training_set, validation_data = test_set, epochs = int(self.ui.txtEpoch.text()))
            self.ui.listViewVeriler.addItem("Eğitilen modelin sonuçları:")
            train_loss=self.historyCNN.history['loss']
            val_loss=self.historyCNN.history['val_loss']
            train_acc=self.historyCNN.history['accuracy']
            val_acc=self.historyCNN.history['val_accuracy']
            self.ui.listViewVeriler.addItem(f'Accuracy: {np.mean(train_acc)})')
            self.ui.listViewVeriler.addItem(f'Loss: {np.mean(train_loss)}')
            self.ui.listViewVeriler.addItem(f'Val Accuracy: {np.mean(val_acc)})')
            self.ui.listViewVeriler.addItem(f'Val Loss: {np.mean(val_loss)}')
            if self.ui.modelSaved.isChecked()==True:
                self.save()
                # self.cnn.save("C:\\Users\\djadi\\OneDrive\\Рабочий стол\\chest_xray\\Result\\ModelFromAraYuz.h5")
        else:self.show_popup()
    
    def save(self):
         
        filePath = str(QFileDialog.getExistingDirectory(self,"h5 file kaydedilecek dosya yerini seçiniz"))
        if filePath == "":
            return
         
        # saving canvas at desired path
        self.cnn.save(filePath+"/ModelFromAraYuz.h5")
    
    def image_siniflandirma(self):
        if self.file[0] != 0:
            for i in ["/NORMAL/", "/PNEUMONIA/"]:
                for image in (os.listdir(self.test_dir + i)):
                    image = plt.imread(self.test_dir + i + image)
                    image = cv2.resize(image, (244, 244))
                    image = np.dstack([image, image, image])
                    image = image.astype("float32") / 255
                    if i == "/NORMAL/":
                        label = 0
                    elif i == "/PNEUMONIA/":
                        label = 1
                    self.test_data.append(image)
                    self.test_labels.append(label)

            self.test_data = np.array(self.test_data)
            self.test_labels = np.array(self.test_labels)
        else: self.show_popup()
    
    def image_prediction(self,new_image_path):
        
        if self.h5file[0]!=0:
            # self.image_siniflandirma()
            history=tf.keras.models.load_model(self.h5file)
            predictions = history.predict(self.test_data)
            test_image = load_img(new_image_path, target_size = (224, 224))
            test_image = img_to_array(test_image)
            #test_image = np.reshape(test_image, (224, 224, 3))
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image / 255.0
            #prediction = model_loaded.predict(test_image)
            test_image_for_plotting = load_img(new_image_path, target_size = (224, 224))
            plt.imshow(test_image_for_plotting)
            if(predictions[0] > 0.5):
                statistic = (1.0 - predictions[0]) * 100
                return self.ui.listViewVeriler.addItems(["Bu görüntü yüzde %.3f %s" % (statistic, " ===> N O R M A L")])
            else:
                statistic = predictions[0] * 100 
                return self.ui.listViewVeriler.addItems(["Bu görüntü yüzde %.3f %s"% (statistic, " ===> P N E U M O N I A")])
        else: self.show_popup()
        
     
 
# call and use the function

    
    def TestingImage(self):
        if self.fname[0]!=0:
            # new_image_path = "/content/chest_xray/train/PNEUMONIA/person1003_virus_1685.jpeg"
            # test_image = load_img(new_image_path, target_size = (224, 224))
            # test_image = img_to_array(test_image)
            # test_image = np.expand_dims(test_image, axis = 0)
            # #test_image = np.reshape(test_image, (1, 224, 224, 3))
            # test_image = test_image / 255

            # #___________________________________________________________________

            # if predictions[0][0] == 0:
            #     prediction = "N O R M A L"
            # else:
            #     prediction = "P N E U M O N I A"

            # print(prediction)
            self.image_prediction(self.fname[0])
        else: self.show_popup()
        
    def AddVeriToListView(self):
        if self.file[0] != 0:
            train_count = glob.glob(self.train_dir+'/**/*.jpeg')
            val_count = glob.glob(self.val_dir+'/**/*.jpeg')
            test_count = glob.glob(self.test_dir+'/**/*.jpeg')
            self.ui.listViewVeriler.addItems(["〰"*30])
            self.ui.listViewVeriler.addItems([f'Training Set has: {len(train_count)} images'])
            self.ui.listViewVeriler.addItems([f'Testing Set has: {len(test_count)} images'])
            self.ui.listViewVeriler.addItems([f'Validation Set has: {len(val_count)} images'])

            sets = ["train", "test", "val"]
            all_pneumonia = []
            all_normal = []

            for cat in sets:
                path = os.path.join(self.file, cat)
                norm = glob.glob(os.path.join(path, "NORMAL/*.jpeg"))
                pneu = glob.glob(os.path.join(path, "PNEUMONIA/*.jpeg"))
                all_normal.extend(norm)
                all_pneumonia.extend(pneu)
            self.ui.listViewVeriler.addItems(["〰"*30])
            self.ui.listViewVeriler.addItems([f"Total Pneumonia Images: {len(all_pneumonia)}"])
            self.ui.listViewVeriler.addItems([f"Total Normal Images: {len(all_normal)}"])
            self.ui.listViewVeriler.addItems(["〰"*30])

            
    def Matrix(self):
        if self.h5file[0]!=0: 
            history=tf.keras.models.load_model(self.h5file) 
            predictions = history.predict(self.test_data)
            conf_m = confusion_matrix(self.test_labels, np.round(predictions))
            self.acc = accuracy_score(self.test_labels, np.round(predictions)) * 100
            self.tn, self.fp, self.fn, self.tp = conf_m.ravel()
            fig, ax = plot_confusion_matrix(conf_mat = conf_m, figsize = (6, 6), cmap = matplotlib.pyplot.cm.Reds)
            plt.show()
        else:self.show_popup()
        
    def KayipMatrisi(self):
        plt.figure(figsize = (10, 5))
        plt.title("Model loss")
        plt.plot(self.historyCNN.history["loss"], "go-")
        plt.plot(self.historyCNN.history["val_loss"], "ro-")
        plt.legend(["loss", "val_loss"])
        plt.show()
    def DogrulukMatrisi(self):
        plt.figure(figsize = (10, 5))
        plt.title("Model accuracy")
        plt.plot(self.historyCNN.history["accuracy"], "go-")
        plt.plot(self.historyCNN.history["val_accuracy"], "ro-")
        plt.legend(["accuracy", "val_accuracy"])
        plt.show()
        
def app():
    app=QtWidgets.QApplication(sys.argv)
    win=window()
    win.show()
    sys.exit(app.exec_())

app()