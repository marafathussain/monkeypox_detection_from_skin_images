# AI to Detect Monkeypox from Digital Skin Images

This is the code repository for our [paper](#cite) titled "[Can Artificial Intelligence Detect Monkeypox from
Digital Skin Images?](https://www.biorxiv.org/content/10.1101/2022.08.08.503193v3)" that used a web-scrapping-based Monkeypox, Chickenpox, Smallpox, Cowpox, Measles, and healthy skin image dataset to study the feasibility of using state-of-the-art AI deep models on skin images for Monkeypox detection  

[PDF Link](https://www.biorxiv.org/content/biorxiv/early/2022/08/09/2022.08.08.503193.full.pdf) | [DOI](https://doi.org/10.1101/2022.08.08.503193)

## Motivation

- The clinical attributes of Monkeypox resemble those of Smallpox, while skin lesions and rashes of Monkeypox often resemble those of other poxes, for example, Chickenpox and Cowpox. 
- These similarities make Monkeypox detection challenging for healthcare professionals by examining the visual appearance of lesions and rashes. - Additionally, there is a knowledge gap among healthcare professionals due to the rarity of Monkeypox before the current outbreak. 
- Motivated by the success of artificial intelligence (AI) in COVID-19 detection, the scientific community has shown an increasing interest in using AI in Monkeypox detection from digital skin images.

## Brief Description of the Method
We test the feasibility of using state-of-the-art AI techniques to classify different types of pox from digital skin images of pox lesions and rashes. The novelties of this work are the following.
- We utilize a database that contains skin lesion/rash images of 5 different diseases, i.e., Monkeypox, Chickenpox, Smallpox, Cowpox, and Measles, as well as contains healthy skin images.
- Our database contains more data for pox, measles, and healthy images scraped on the Web (i.e., 804 images), before augmentation, compared to other similar databases.
- We tested the disease classification power of seven state-of-the-art deep models from digital skin images. We tested the disease classification performance of ResNet50, DenseNet121, Inception-V3, SqueezeNet, MnasNet-A1, MobileNet-V2, and ShuffleNet-V2.
- We performed 5-fold cross-validation tests for each of the AI deep models to more comprehensively analyzing our findings.


### Code Description
We shared our codes and example inference jupyter notbook. 
- Code named "densenet121_5f.py" implements a 5-fold cross-validation using DenseNet121 deep network.
- Code named "inception_v3_5f.py" implements a 5-fold cross-validation using Inception-V3 deep network.
- Code named "mnasnet1_0_5f.py" implements a 5-fold cross-validation using MNasNet-A1 deep network.
- Code named "mobilenet_v2_5f.py" implements a 5-fold cross-validation using MobileNet-V2 deep network.
- Code named "resnet50_5f.py" implements a 5-fold cross-validation using ResNet50 deep network.
- Code named "shufflenetv2_1_5f.py" implements a 5-fold cross-validation using ShuffleNet-V2-1× deep network.
- Code named "squeezenet1_1_5f.py" implements a 5-fold cross-validation using SqueezeNet deep network.

In addition, "pretrained_models_test_run.ipynb" contains codes for finetuning deep models on monkeypos digital skin image data. Also "monekypox_cv_test_run.ipynb" contains the inference and results generation codes. 

<a name="cite"></a>
### Cite
```bibtext
@article{hussain2022can,
  title={Can artificial intelligence detect monkeypox from digital skin images?},
  author={Islam, Towhidul and Hussain, Mohammad Arafat and Chowdhury, Forhad Uddin Hasan and Islam, BM Riazul},
  journal={BioRxiv},
  pages={2022--08},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
