# -*- coding:utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob


class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path="./data/train/image", label_path="./data/train/label",
                 test_path="./data/test", npy_path="./npydata", img_type="png"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):
        i = 0

        imgs = glob.glob(self.data_path+"/*."+self.img_type)#列出目錄下所有圖片
        
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for x in range(len(imgs)):
            imgpath = imgs[x]
            #print(imgpath)
            pic_name = imgpath.split('\\')[-1] # 抓出照片名稱
            labelpath = self.label_path + '\\' + pic_name #把原本路徑加照片名稱組成每張label路徑

            img = load_img(imgpath, grayscale=False, target_size=[512, 512])
            label = load_img(labelpath, grayscale=True, target_size=[512, 512])
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label

            i += 1

        #print('test')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)

    def create_test_data(self):
        i = 0

        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        print(imgs)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        testpathlist = []

        for imgname in imgs:
            temp = imgname.split('\\')[-1]
            #print(temp)
            testpath = self.test_path+'/'+temp
            print(imgname)
            testpathlist.append(testpath)
            img = load_img(testpath, grayscale=False, target_size=[512, 512])
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1

        txtname = './results/pic.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)


    def load_train_data(self):

        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1  # 白
        imgs_mask_train[imgs_mask_train <= 0.5] = 0  # 黑
        return imgs_train, imgs_mask_train

    def load_test_data(self):

        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test

if __name__ == "__main__":
    mydata = dataProcess(512, 512)
    mydata.create_train_data()
    mydata.create_test_data()
