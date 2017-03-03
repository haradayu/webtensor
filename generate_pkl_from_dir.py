#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import cv2
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def generate_pkl_from_dir(path, output_path, out_dir = "",size = (28, 28), val_rate = 0, test_rate = 0, random_state = 0, gray_mode = False):
    if out_dir != "" and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if gray_mode == True:
        gray_flag_cv2 = cv2.IMREAD_GRAYSCALE
    else:
        gray_flag_cv2 = cv2.IMREAD_COLOR
    #読み込む拡張子
    ext_list = [".jpg", ".jpeg", ".png"]
    # データを入れる配列
    train_image = []
    train_path = []
    train_label = []
    val_image = []
    val_label = []
    val_path = []
    test_image = []
    test_label = []
    test_path = []
    label_name = []
    # ファイルを開く
    subdir_list = []
    #サブフォルダの一覧を取得
    for subdir in sorted(os.listdir(path)):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            subdir_list.append(subdir_path)
            label_name.append(subdir)
    #サブフォルダごとの画像を読み込み
    for i,subdir_path in enumerate(subdir_list):
        tmp_path = []
        tmp_image = []
        tmp_label = []
        print subdir_path
        for img_path in tqdm(sorted(glob.glob(os.path.join(subdir_path, "*.*")))):
            if os.path.splitext(img_path)[1] not in ext_list:
                continue
            tmp_path.append(img_path)
            img = cv2.imread(img_path, gray_flag_cv2)
            img = cv2.resize(img, size)
            # 一列にした後、0-1のfloat値にする
            tmp_image.append(img.flatten().astype(np.float32)/255.0)
            tmp = np.zeros(len(subdir_list))
            tmp[i] = 1  
            tmp_label.append(tmp)
        
        if val_rate + test_rate == 0:
            #評価用とテスト用にわけない場合は全部訓練用
            train_image.extend(tmp_image)
            train_path.extend(tmp_path)
            train_label.extend(tmp_label)
            continue
        #訓練データとその他に分割
        train_and_valtest = train_test_split(zip(tmp_image, tmp_path), tmp_label, test_size = (val_rate + test_rate), random_state = random_state)
        # print train_and_test[1]
        # print train_and_test[3]
        train_image.extend([x[0] for x in train_and_valtest[0]])
        train_path.extend([x[1] for x in train_and_valtest[0]])
        train_label.extend(train_and_valtest[2])
        valtest_image = train_and_valtest[1]
        valtest_label = train_and_valtest[3]
        #評価用データとテスト用データに分割
        val_and_test = train_test_split(valtest_image, valtest_label, test_size =(test_rate) /(val_rate + test_rate), random_state = random_state)        
        val_image.extend([x[0] for x in val_and_test[0]])
        val_path.extend([x[1] for x in val_and_test[0]])
        test_image.extend([x[0] for x in val_and_test[1]])
        test_path.extend([x[1] for x in val_and_test[1]])
        val_label.extend(val_and_test[2])
        test_label.extend(val_and_test[3])
    #arrayからnumpy形式に変換
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)
    val_image = np.asarray(val_image)
    val_label = np.asarray(val_label)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    #pikle形式で保存
    with open(os.path.join(out_dir,output_path), "wb") as f:
        pickle.dump([train_image, train_label, val_image, val_label,test_image, test_label, label_name],f)
    
    #訓練用，評価用，テスト用それぞれのファイル名をテキストに保存
    with open(os.path.join(out_dir, path + "_train.txt"), "w") as f:
        for img_path in sorted(train_path):
            f.write(img_path + "\n")
            
    if val_rate == 0:
        return
    with open(os.path.join(out_dir, path + "_val.txt"), "w") as f:
        for img_path in sorted(val_path):
            f.write(img_path + "\n")  
            
    if test_rate == 0:
        return
    with open(os.path.join(out_dir, path + "_test.txt"), "w") as f:
        for img_path in sorted(test_path):
            f.write(img_path + "\n")
            
            
generate_pkl_from_dir("testing", "out.pkl", out_dir = "mnist" ,val_rate = 0.2, test_rate = 0.1, gray_mode = True)
