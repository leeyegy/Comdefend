# run in env torch1.1
for attack in FGSM PGD Momentum DeepFool
do
	for epsilon in  0.01569
	do
		python main_cifar10.py --attack_method $attack --epsilon $epsilon --set_size 10000 | tee ./logs/CIFAR10_$attack\_$epsilon.txt
	done
done
