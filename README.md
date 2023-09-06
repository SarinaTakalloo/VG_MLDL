### This repository is based on CosPlace project available in the link below:

[https://github.com/gmberton/CosPlace.git].
--------------------------------------------------------------------------------------------------------------------------------------------

In this repository all the training and testing are done using sf_xs and tokyo_xs.

1) sf_xs size is 1,22 GB on disk and 103.869 items.

2) tokyo_xs size is 164,6 MB on disk and 13.090 items.

In addition two backbones and data augmentation is included:

| backbones       | augmentation        |
  :---            |         ---:  
| Wide ResNet50-2 | use_horizontal_flip | 
| densenet121     | degrees             |

--------------------------------------------------------------------------------------------------------------------------------------------
#### Results of our tests on the datasets are shown as below:


Best models results: Recall@1 / Recall@5

SF-XS val(R@1 / R@5)|SF-XS(R@1 / R@5)| Tokyo-XS(R@1 / R@5)
 :---               |     :---:      |     ---:   
|  56.4 / 69.4      | 20.9 / 32.6    | 34.0 / 52.4
--------------------------------------------------------------------------------------------------------------------------------------------

Results of different parameter for data augmentation:

 Model               | SF-XS val(R@1 / R@5)|SF-XS(R@1 / R@5)| Tokyo-XS(R@1 / R@5)
 :---                |     :---:           |     :---:      |     ---:   
|degree=5            | 56.6 / 69.5         | 22.2 / 35.0    | 33.3 / 54.3
|degree=10           | 56.0 / 68.7         | 22.3 / 36.7    | 30.8 / 56.2 
|horizontal_flip=true| 55.9 / 69.6         | 19.7 / 32.6    | 31.1 / 54.3 
--------------------------------------------------------------------------------------------------------------------------------------------

Comparison between ResNet18 and VGG16: Recall@1 / Recall@5
 Model    | SF-XS val(R@1 / R@5)|SF-XS(R@1 / R@5)| Tokyo-XS(R@1 / R@5)
 :---     |     :---:           |     :---:      |     ---:   
|ResNet18 | 56.4 / 69.4         | 20.9 / 32.6    | 34.0 / 52.4
|VGG16    | 70.0 / 80.8         | 36.5 / 50.7    | 51.7 / 74.0 
--------------------------------------------------------------------------------------------------------------------------------------------

Comparison of Performance between Adam and AdamW Optimizers
Dataset    | Default (Adam lr e-5)(R@1 / R@5) | AdamW (lr e-5)(R@1 / R@5)
:---       |        :---:                     |     ---:    
SF-XS val  | 56.4 / 69.4                      | 56.2 / 69.3 
SF-XS test | 20.9 / 32.6                      | 19.6 / 32.5 
Tokyo-XS   | 34.0 / 52.4                      | 33.3 / 53.3 
--------------------------------------------------------------------------------------------------------------------------------------------

![photo_2023-09-05 14 44 44](https://github.com/SarinaTakalloo/VG_MLDL/assets/98056551/fb381428-e48a-48cf-a095-8c07cd5707dd)

![photo_2023-09-05 14 44 42](https://github.com/SarinaTakalloo/VG_MLDL/assets/98056551/328a0279-9a8a-476a-8d40-eaec550b3495)


--------------------------------------------------------------------------------------------------------------------------------------------
### Our tests and pre-trained models are available at this link:

[https://drive.google.com/drive/folders/1--7FrsGjZBLAnE02On0ekd9endqXB3lE?usp=sharing]
--------------------------------------------------------------------------------------------------------------------------------------------


