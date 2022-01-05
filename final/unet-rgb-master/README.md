# final project.py
檔案結構  
資料下載 https://drive.google.com/file/d/1koMCHlZj7tNuP61rFAsCuuTIe5adujdr/view?usp=sharing     
包含data、npydata、results
```
unet-rgb-master/
|-- data.py
|
|-- data/
|   |-- train/
|   |   |-- image/
|   |   |   |-- 1.png
|   |   |   |-- ..
|   |   |   |-- ..
|   |   |   |-- 32.png
|   |   |-- label/
|   |   |   |-- 1.png
|   |   |   |-- ..
|   |   |   |-- ..
|   |   |   |-- 32.png
|   |-- test/
|   |   |-- 0.png
|   |   |-- ..
|   |   |-- ..
|   |   |-- 5.png
|   |-- results/
|-- npydata/
|   |-- imgs_mask_train.npy
|   |-- imgs_test.npy
|   |-- imgs_train.npy
|-- results/
|   |-- pic.txt
|   |-- imgs_mask_test
|-- test2mask2pic.py
|-- unet.py
|-- unet.hdf5
|-- unet_model.hdf5
|-- requirements.txt
|-- README

```

### data preprocess
```
python data.py
```

### train
```
python unet.py
```

### inference
```
python test2mask2pic.py
```


