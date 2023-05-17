# VG
## Classification for Retrieval in Image Geolocalization
### 1) OVERVIEW:



 Visual geo-localization (VG), also called Visual Place Recognition (VPR), is the task of coarsely finding the geographical position where a given photograph was taken. This task is commonly addressed as an image retrieval problem: given an unseen image to be localized (query), it is matched against a database of geo-tagged images that represent the known world. The N-top retrieved images, with their geo-tag (typically GPS coordinates), provide the hypothesis for the query's geographical location.
In this project you will work with an innovative approach that casts the problem as Classification, at training time. For inference, the extracted features are used for retrieval as usual.
The retrieval is performed as a k Nearest Neighbour search in a learned embedding space that well represents the visual similarity of places. Namely, each image is passed through a network composed of a feature extraction backbone and a head that aggregates or pools the features to create a global descriptor of the image. 
In earlier work the training was performed using a contrastive learning approach. The model learns to extract descriptors that are discriminative of locations. The similarity search is then implemented as a pairwise comparison among descriptors, e.g. using a cosine similarity. In the new presented approach, at train time the network divides the geographical area to classify into squares, distinguishing inside each square the heading (e.g. spatial orientation), and is trained to classify the location of the images.
In this project, you will become familiar with the task of visual geo-localization and with how a visual geo-localization system works. You will start by experimenting with the baselines using a codebase provided by us.Subsequently you will focus on improving one or more aspects of the system of your choice, from robustness to illumination changes (e.g. pictures taken at night), different perspectives and occlusions.  We already provide you with the code to implement the baseline, so that you do not have to waste time in re-implementing them; in this way you can focus on understanding the code, and then try to tackle real research problems. If you obtain any significant results and if you are interested you can bring the project to a publication to a computer vision conference or scientific journal.
### 2) DATASETS



We provide you with multiple datasets to experiment with:
San Francisco eXtra Small (SF-XS): a subset of the SF-XL dataset [19]. This is used for training (SF-XS train), validation (SF-XS val) and testing (SF-XS test).
Tokyo eXtra Small (Tokyo-XS): a subset of the Tokyo 24/7 [16] dataset. This is used only for testing.

The reason for using two test sets is to understand how your changes to the model affect the geolocalization for different datasets: for example, some extensions might improve results for Tokyo-XS, but worsen results on SF-XS (test).
Note that validation and test datasets have two directories, one for database and one for the queries: this is because you need both for evaluation. On the other hand at training time the code uses a single set of images (although some older papers [1] used database and queries also for training).
The datasets are available [at this link](https://drive.google.com/drive/folders/1Ucy9JONT26EjDAjIJFhuL9qeLxgSZKmf)

### 3) CODE



The code is available at this [GitHub repo](https://github.com/gmberton/CosPlace). Leaving a Star on the repo is highly appreciated. [Here](https://colab.research.google.com/drive/13u3FBPTKBKJ470TtrW7eYPJo-7l0T3lE?usp=share_link) is also a notebook with an example on how to use the code

### 4) STEPS


**a) Study the literature and get familiar with the task**
As a preliminary step to get familiar with the task, start by understanding what is image retrieval, and how it is used for visual geolocalization / visual place recognition. At first use Google, ChatGPT and YouTube to find relevant resources. Then read some papers to get an idea on how the task is solved. Understand what are the challenges (e.g. night-time photos are difficult to localize!), and how they are solved. You can also refer to this survey [3] for a broader overview of the task and its evolution through the years. After you have an idea about the task, carefully understand CosPlace [20], which will serve as a baseline for the next steps.


**b) Time to play with the code!**

Download the datasets, download the code, and run your first training! Your neural network will learn how to solve the task, perhaps it will be able to geolocalize images even better than you!
Once you are familiar with the theory of CosPlace, you can start to run some experiments to better understand how the training procedure works. You will use a ResNet-18 pre-trained on ImageNet and GeM pooling [2], as already implemented in the code.
Before getting started with the code, make sure to understand how the datasets are built, how dense they are, which labels are available, plot some diagram with the distribution of the labels and visualize some images, and study the training code as well. The SF-XS train set is quite small: to keep it small, we filtered away most images from SF-XL, and we kept images only from one group, at most 10 images per class, and using only one heading direction per class.
### IMPORTANT - Before training change the `queries_v1` [here](https://github.com/gmberton/CosPlace/blob/main/train.py#L60) and [here](https://github.com/gmberton/CosPlace/blob/main/eval.py#L40) to `queries` in the code. Also comment the line 87 and 88 of train_dataset.py [here](https://github.com/gmberton/CosPlace/blob/main/datasets/train_dataset.py#L87)
Hint: use the following command to run a training with default values:
!python train.py --dataset_folder /path/to/dataset --groups_num 1
At test time, the query image is deemed correctly localized if at least one of the top N retrieved database images is within d = 25 meters from the ground truth position of the query. To assess these baselines, you will have to report the values of recall@N (for N=1,5), that is the percentage of correctly recognized queries (Recall) using the N top retrieved images. Are the results different on SF-XS val from SF-XS test? Why?
Have a look at the images in the datasets, and make sure that you understand why the results are different on the two sets! What are the main differences between SF-XS val and SF-XS test?


**c) Use the trained model to test on Tokyo-XS**

Do you really need to train your model again for this step? The answer is no, the weights of the trained models are saved in the logs, you can resume those to test with your trained model! Are the results different on Tokyo-XS than on SF-XS?
Have a look at the images in Tokyo-XS, what are the main differences between Tokyo-XS and SF-XS test?
At test time, the query image is deemed correctly localized if at least one of the top N retrieved database images is within d = 25 meters from the ground truth position of the query. You will have to report the values of recall@N (for N=1,5), the percentage of queries for which at least one of the first N predictions is within a 25 meters distance from the query.
Your first table should look something like this


| SF-XS val  | SF-XS test |Tokyo-XS |
| -----------| -----------|---------|
| R@1/R@5    | ....       | ....    |


**d) Visually analize the results**


There is a parameter in the code that you can pass to the training script that will save some queries and their predictions (i.e. the images from the database that the model thinks are most similar to the query). Find it and use it! How do the predictions look? Does your model make any strange mistakes?

**e) Now it’s time to do some real research!**

Are you ready to be a Deep Learning researcher? Then choose at least one of the following steps and work on it! Always make sure to understand what you are doing, make some guesses on how your changes could influence the neural net, and then launch your experiment to see if your guesses were correct. Also, don’t forget to visually inspect the predictions. When applying data augmentation, visually inspect the augmented images. The objective of your contributions should be to improve results or speed up training. Test your trained models on all test datasets.

 _Data augmentation_
 
How good are the predictions? Do you notice any recurring issues? Does your model always makes the same kind of mistakes?
Try to add some other types of data augmentation (in the code there is ColorJitter and RandomResizeCrop already), and see if the results can improve!
But be careful, the current version of data augmentation is performed on GPU, and a loop is used to make sure that each image in the batch is augmented separately. You can implement it in the same fashion, or you can use some cool libraries to do data augmentation, like Nvidia DALI or Albumentations (Google them, it is full of tutorials online!)
If you choose this step, make sure to perform a thorough set of experiments, do not just try one single augmentation.
_Backbone_

What backbone have you used until now? Is it a strong backbone? Can you find some other backbones that perform better (ideally something that can train faster and have better results)? Look for the “timm” library for PyTorch, they have plenty of models that you can use as backbones!

_Aggregation_

The last part of the model (aggregation) is a pooling layer and a fully connected (i.e. linear) layer. Are there better alternatives? Try to implement the MLP-mixer style aggregation presented in [22], or any other pooling / aggregation methods (for example some attention layers).

_Self-supervised training_

In this library [link](https://kevinmusgrave.github.io/pytorch-metric-learning/) you find implemented several self-supervised losses like VicReg, and many others. To exploit these self-supervised strategies, you need to define some augmentations, so that then you can ask the network to have embeddings robust to that kind of augmentation. Therefore augmentations design, and the choice of the loss, are the most important design choices. Regarding augmentations, you can try RandomPerspective, cropping, random erasing; design the augmentation based on what you think is useful for the task.

_Domain Adaptation_

Visual Geolocalization is by nature a cross-domain task, because the queries might come from a different domain from database (usually the database is collected from Google StreetView images, while the queries are photos taken with a phone, sometimes at night or from difficult viewpoints). Domain Adaptation can be helpful in these cases.
You can use the test queries as target domain, and use them for domain adaptation.

_Optimizer & Schedulers_

As the base parameter for training, the default is the Adam optimizer with LR 1e-5. Try to experiment with different recently proposed optimizers (beyond SGD) available in torch, like AdamW, ASGD. Try to find the better LR, weight decay parameters and understand what happens changing them. You can also try different schedulers like ReduceLrOnPlateau, or CosineAnnealingLR. These should be especially useful if you decide as well to swap the backbone moving to Transformer-based ones.

### 5) DELIVERABLES
To conclude the project you will need to:
- Deliver PyTorch scripts for the required steps as a zip file.
- Write a complete pdf report following a standard paper format. The report should contain a brief introduction, a related works section, a methodological section for describing the algorithms you are going to use, an experimental section with all the results and discussions. End the report with a brief conclusion.
**Some questions you should be able to answer in the end:
**What is contrastive learning and how it relates to image retrieval?
How does CosPlace group images into classes?

### 6) REFERENCES
[1] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla and J. Sivic, NetVLAD:  CNN  architecture  for weakly  supervised  place  recognition, TPAMI 2018
[2] F. Radenovic, G. Tolias, and O. Chum, Fine-tuning CNN Image Retrieval with No Human Annotation, TPAMI 2018.
[3] C. Masone and B. Caputo, A survey on deep visual place recognition, IEEE Access 2021
[4] L. Liu, H. Li, and Y. Dai, Stochastic Attraction-Repulsion Embedding for Large Scale Image Localization, ICCV 2019.
[10] H. J. Kim, E. Dunn, and J.-M. Frahm, Learned contextual feature reweighting for image geo-localization,  CVPR 2017
[13] X. Wang, R. Girshick, A. Gupta and K. He, Non-local neural networks, CVPR 2018.
[14] F. Warburg, S. Hauberg, M. Lopez-Antequera, P. Gargallo, Y. Kuang and J. Civera, Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition, CVPR 2020

[16]A. Torii, R. Arandjelovi ́c, J. Sivic, M. Okutomi, and T.Pajdla. 24/7 place recognition by view synthesis. IEEETransactions on Pattern Analysis and Machine Intelligence,40(2):257–271, 2018.

[17]Yang, Min, Dongliang He, Miao Fan, Baorong Shi, Xuetong Xue, Fu Li, Errui Ding and Jizhou Huang. “DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features.” 2021 IEEE/CVF International Conference on Computer Vision (ICCV) (2021): 11752-11761.

[18] Cao, Bingyi, Andre F. de Araújo and Jack Sim. “Unifying Deep Local and Global Features for Image Search.” ECCV (2020).

[19] Berton, G. and Mereu, R. and Trivigno, G. and Masone, C. and Csurka, G. and Sattler, T. and Caputo, B. “Deep Visual Geo-localization Benchmark.” CVPR (2022).

[20] Berton, G. and Masone, C. and Caputo, B. “Rethinking Visual Geo-localization for Large-Scale Applications.” CVPR (2022).

[21] Ali-bey, Amar, Brahim Chaib-draa, and Philippe Giguère. "GSV-Cities: Toward appropriate supervised visual place recognition." Neurocomputing 513 (2022): 194-203.

[22] Ali-bey, Amar, Brahim Chaib-draa, and Philippe Giguère. "MixVPR: Feature Mixing for Visual Place Recognition." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.
