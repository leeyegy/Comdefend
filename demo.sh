# run in env torch1.1
for attack in  PGD Momentum FGSM
do
	for epsilon in  0.00784 0.03137 0.06275
	do
		python main_cifar10.py --attack_method $attack --epsilon $epsilon --set_size 1000 | tee ./logs/tiny_imagenet/$attack\_$epsilon.txt
	done
done
for attack in  DeepFool STA CW NONE
do
	for epsilon in  0.0
	do
		python main_cifar10.py --attack_method $attack --epsilon $epsilon --set_size 1000 | tee ./logs/tiny_imagenet/$attack\_$epsilon.txt
	done
done
