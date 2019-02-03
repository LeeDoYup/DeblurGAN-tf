# DeblurGAN-tf
DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks [paper](https://arxiv.org/abs/1711.07064)  
This repository is tensorflow (tf) implementation of DeblurGAN.

DeblurGAN removes blur filter in a image and make the image sharp, as follow:  
<img src='assets/animation3.gif' width='400px'/> <img src='assets/animation4.gif' width='400px'/>

# First, Download GOPRO Dataset
DeblurGAN model is trained using GOPRO dataset.  
To train the model, download the dataset from website [download](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view)   

Unzip the dataset wherever you want, and remember the (data_path).  

# Train Model
You can train DeblurGAN model from sratch, using GOPRO dataset.
```
python3 train.py --is_training --data_path (data_path)
```

When the model training ends, you can identify the results of Debluring GOPRO dataset.  
The results are saved in `./test_result/`

# Deblur your own Image
After training the model, you can deblur your own images using the trained model.  





# Now the model is being Trained.
