## Running ESRGAN

```
1. pip install numpy opencv-python
2. mkdir InputImages
   (this is where all the input images will reside in)
3. mkdir OutputImages
   (this is where the model will generate all the images)
4. mkdir models
   (save the .pth checkpoint file here)
5. python esrgan.py
```

- Download [RRDB_ESRGAN_x4.pth](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) and place it in the `models` directory as mentioned above in step 4.
- Credits : [ESRGAN](https://github.com/xinntao/ESRGAN)
