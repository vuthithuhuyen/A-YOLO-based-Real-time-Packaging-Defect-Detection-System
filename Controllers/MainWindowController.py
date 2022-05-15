import os
import time
from os import path

import numpy as np
from joblib import dump, load
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QMainWindow, QMessageBox

from Helper.FileHelper import sliderChangeValue
from Helper.JSVShowinginThread import stop_jsv_witing_thread
from Helper.OpenCVHelper import isVideo, RawImageToArray
from Helper.PyQTHelper import QLabelDisplayImage, showDialog, showErrDialog
from Helper.SVMHelper import SVMTrainer

from Tool_Model import GlobalVariables
from Tool_Model.GlobalVariables import display_rescale, train_class, thres1, thres2, trained_svm, currentVideoFile
from pathlib import Path
import pandas as pd

from Tool_Model.RunningTime import MyRunningTime

# Bấm vào danh sách file trên list. Nếu là file ảnh thì hiển thị
from detect import GenerateCompareVideo
from detect_new import detection
def FileClick(filename):
    if not isVideo(filename):
        print(f'{filename} is not a video file')
        return
    try:
        GlobalVariables.currentVideoFile = filename
    except Exception as e:
        print(e)


# Stop Generate Video:
def StopGenerateVideo():
    print('STOP GENERATE VIDEO!')
    GlobalVariables.stopVideo = True
    cv2.destroyAllWindows()
    stop_jsv_witing_thread()


# analyze obect to elert from text
def AnalyzeObjectToAlertText(txtObjectAlert):
    try:
        GlobalVariables.alert_dict = {}
        object_group = txtObjectAlert.split(',')

        for obj in object_group:
            try:
                GlobalVariables.alert_dict[(obj.split(':')[0]).strip()] = int(obj.split(':')[-1])
            except:
                pass

        print(f'Alert dict: {GlobalVariables.alert_dict}')

    except Exception as e:
        print(f'error on: {e}')
    finally:
        pass


# Generate video data
def GenerateVideo(mainWindow: QMainWindow, filePath):
    try:
        GlobalVariables.stopVideo = False

        # Preprocessing for sending email
        GlobalVariables.flag_send_email = mainWindow.checkbox_sendEmail.isChecked()
        object_alert = mainWindow.txtObjectAlert.toPlainText()
        AnalyzeObjectToAlertText(object_alert)

        isLocalfile = mainWindow.checkbox_LocalFiles.isChecked()

        save_video = mainWindow.checkbox_saveVideo.isChecked()
        rescale = mainWindow.spin_rescale.value()


        if not isLocalfile:
            GlobalVariables.currentVideoFile = mainWindow.txtVideoURL.toPlainText()

        print(f'Video file: {GlobalVariables.currentVideoFile}')

        # GenerateCompareVideo(source=GlobalVariables.currentVideoFile, weights='best.pt',
        #                      nosave=not save_video, rescale=rescale)

    except Exception as e:
        print(f'Error on GenerateVideo: {e}')


# huấn luyện dữ liệu dựa trên thư mục đã chọn
def TrainingInputData_click(training_path, statusLabel: QLabel):
    try:
        if not (os.path.isdir(training_path)):
            print(f'{training_path} is not exsits!')
            return
        # Duyệt thư mục, lấy danh sách các file ảnh để chuẩn bị huấn luyện
        print('Start reading all files... Please wait...')
        p = Path(training_path)
        all_files = []
        for i in p.rglob('*.jpg'):
            msg = f'Reading file to queue list... {i.name}'
            print(msg)
            statusLabel.setText(msg)
            image_class = i.name.split('_')[0]
            if not (image_class in train_class):
                continue
            all_files.append((image_class, i.name, i.absolute(), time.ctime(i.stat().st_ctime)))
            i.absolute()
        columns = ["Image_Class", "File_Name", "Full_Path", "Created"]
        # Tạo data frame chứa danh sách các file để huấn luyện.
        df = pd.DataFrame.from_records(all_files, columns=columns)
        train_data, train_label = ListFileInFrame2TrainData(df)
        executeTime = MyRunningTime()
        print('Start training SVM...')
        statusLabel.setText('Start training SVM...')
        # Train SVM model
        GlobalVariables.trained_svm = SVMTrainer(train_data, train_label)
        print(GlobalVariables.trained_svm)
        statusLabel.setText('Trained SVM model')
        showDialog("Information", "Finish training SVM model.")
        executeTime.CalculateExecutedTime()
    except Exception as e:
        print(e)


# Đọc dữ liệu file từ danh sách trong data frame -> Chuyển vào train_data, train_label
def ListFileInFrame2TrainData(df: pd):
    list_data, listLabel = [], []
    executeTime = MyRunningTime()

    for index, row in df.iterrows():
        print(f'Reading image... {row[1]}')
        label = row[0]
        img_array = RawImageToArray(row[2], thres1, thres2)

        listLabel.append(label)
        list_data.append(img_array)

    features = len(list_data[0])
    train_data = np.empty((0, features), int)
    train_label = np.array(listLabel)

    count = 0
    for row in list_data:
        try:
            count += 1
            row = row.reshape(-1, features)
            train_data = np.append(train_data, row, axis=0)
        except Exception as e:
            print(f'Error at index {count} shape: {row.shape}')
            print(e)

    executeTime.CalculateExecutedTime()
    print(f'finish reading file in {executeTime.runningTime}')
    return train_data, train_label


# Bấm nút Dự đoán <Predict>
def PredictClick(filename, lblResult: QLabel):
    try:
        if not isVideo(filename):
            print(f'{filename} is not an image file')
            showErrDialog("Error", "Please select an image")
            return

        if GlobalVariables.trained_svm is None:
            print('No trained SVM model')
            showErrDialog("Error", "No trained SVM model")
            return
        predicted_label = SVMPrediction(filename)
        print(f'predicted: {predicted_label[0]}')
        lblResult.setText(predicted_label[0])
    except Exception as e:
        print(e)


# Dự đoán phân lớp. Prediction
def SVMPrediction(filename):
    try:
        img_array = RawImageToArray(filename, GlobalVariables.thres1, GlobalVariables.thres2)
        input_test = img_array.reshape(-1, len(img_array))
        predicted_label = GlobalVariables.trained_svm.predict(input_test)
        return predicted_label
    except Exception as e:
        print(e)
        return "unknow"


# Load trained model
def LoadTrainedModel_click(self: QMainWindow):
    try:
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "", "SVM trained file (*.svm)")
        if not (os.path.isfile(fileName)):
            return

        GlobalVariables.trained_svm = load(fileName)
        showDialog("Load trained SVM model", f"{fileName} was loaded successful")

        if GlobalVariables.trained_svm is None:
            showErrDialog("Error", "Failed loading SVM")

    except Exception as e:
        print(e)
