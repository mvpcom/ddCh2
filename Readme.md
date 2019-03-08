Programming by Mojtaba Valipour @ SUTest-V1.0.0, [vpcom.ir](http://vpcom.ir/)
Copyright 2019
Title: Deep Hierarchical Persian Text Classification based on hdlTex

# Information about the conda environment (Anaconda)
Environment: hdlTex, vpcomDesk -> hdlTex.yml (conda list --explicit > hdlTex.yml)

Python:3.5.6
Tensorflow: 1.10.0
Keras: 2.2.2
Pandas: 0.23.4
nltk: 3.3.0
numpy: 1.15.2
Cuda:9.0

GPU: Geforce GTX 1080
CPU: Intel® Core™ i7-2600K CPU @ 3.40GHz × 8 
RAM: 12GB
OS: Ubuntu 16.04 LTS 64-bit


# Main Objective
You can find the main objective as follows in the Persian Language:
Challenge2\_DivarDataset\_DataDays, Sharif University
بخش اول
عنوان: پیش بینی دسته بندی
امتیاز: ۳۰۰۰ امتیاز
توانایی: یادگیری ماشین و تحلیل متن
مسئله: پیشبینی دسته بندی آگهی از روی سایر ویژگی های آن

توصیف: در این بخش شما یک دیتاست شامل ۲۰۰ هزار سطر دانلود میکنید که هر سطر حاوی اطلاعات مربوط به یک آگهی است. شما باید دسته بندی سلسله مراتبی هر آگهی را به دست آورید و در قالب یک فایل csv که شامل ۲۰۰ هزار سطر و سه ستون cat1, cat2, cat3 است آپلود کنید.
ملاحظه مهم: ساختار پاسخ باید دقیقا به شکل اشاره شده باشد. ضمنا تمام دسته ها باید به همان شکلی که در دیتاست Train قرار دارد باشد. یک نمونه از پاسخ مطلوب در این فایل فایل پیوست شده است. 


# Hints: 
1. dataDaysChallenge2-Github.ipynb is only for your reference to see how I prepared the main code. Some parts are not compatible with the recent changes!
2. dataDaysChallenge_BIGNet.py is the main code and all the other files are here for your reference only!
3. Make sure you have enough permission and free storage!
4. "preProcessFlag = True" should be true for the first run

# Configuration Example:
You have to change the config vars based on your need:

Config Results on Test set: 
All Categories Acc:  0.9433751213541007  Cat1 Acc:  0.98198683044194  Cat2 Acc:  0.9710966189692288  Cat3 Acc:  0.9520493014224811

```python
epochs = 15; # Number of epochs to train the main model
level2Epochs = 25; # Number of epochs to train the level 2 models
level3Epochs = 40; # Number of epochs to train the level 3 models
MAX_SEQUENCE_LENGTH = 100; # Maximum sequance lentgh 500 words
MAX_NB_WORDS = 55000; # Maximum number of unique words
EMBEDDING_DIM = 300; # Embedding dimension you can change it to {25, 100, 150, and 300} but need the fasttext version in your directory
batch_size_L1 = int(3048/2); # batch size in Level 1
batch_size_L2 = int(3048/2); # batch size in Level 2
batch_size_L3 = int(3048/2); # batch size in Level 3
L1_model = 2; # Model Type: 0 is DNN, 1 is CNN, and 2 is RNN for Level 1
L2_model = 2; # Model Type: 0 is DNN, 1 is CNN, and 2 is RNN for Level 1
L3_model = 2; # Model Type: 0 is DNN, 1 is CNN, and 2 is RNN for Level 1
rnnType = 4; # RNN model, 0 GRU, 1 Conv + LSTM, 2: RNN+DNN 3: Attention, 4: Big
trainingBigNetFlag = True; # one Model for all levels (allInONE ;P), Other Flags will be False automatically
testBigNetFlag = True; # one Model for all levels, Other Flags will be False automatically
```

# Run:
```
source activate hdlTex;
python dataDaysChallenge_BIGNet.py
```

# Inputs:
1. "./data/divar_posts_dataset.csv" # original dataset path, train set
2. "./data/phase_2_dataset.csv" # phase 2 dataset path, test set
3. './fastText/*.vec'

# Outputs:
1. './wordDict.json' # where to save the extracted words dictionary 
2. './dataset/' # where to export processed files for later usage
3. './dataChallenge/' # where to save processed files for phase2 Dataset
4. './resultsChallenge2.csv' # where to save results
5. './resultsChallenge2Inputs.csv' # where to save results and inputs
6. './resultsChallenge2FixLevels.csv' # where to save fixed results, generally better performance
7. './resultsChallenge2FixLevelsALL.csv' # Check all the samples hierarchy (L1,L2,L3)
8. './table.html' # where to save all the results alongside inputs for visual judements

# RESOURCES:
'''
1- [HDLTex](https://github.com/kk7nc/HDLTex)
2- [Glove](https://nlp.stanford.edu/projects/glove/)
3- [CafeBazaar Persian Divar Dataset](https://research.cafebazaar.ir/visage/divar_datasets/)
4- [HDLTex: Hierarchical Deep Learning for Text Classification](https://arxiv.org/abs/1709.08267)
5- [FastText Embedding Vectors](https://fasttext.cc/docs/en/crawl-vectors.html)
6- [Persian NLP](https://github.com/hadifar/PNLP)
7- [Datadays 2019](https://datadays.ir)
8- [Keras Attention Mechanism](https://github.com/philipperemy/keras-attention-mechanism)
'''


