# Comdefend
The code for CVPR2019 (ComDefend: An Efficient Image Compression Model to Defend Adversarial Examples)
[paper](https://arxiv.org/abs/1811.12673)

## Environmental configuration
tensorflow>=1.1 </br>
python3 </br>
canton(pip install canton) </br>

## In addition
This repo is basically for testing in dataset CIFAR10.
* demo_com.sh : using comodefend to get compression data from adv_data and then save it into h5 file.
* demo.sh : following demo_com.sh, it tests accuracy based on compression data which is saved in a specific h5 file. 
