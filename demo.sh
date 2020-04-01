# run in env torch1.1
python main_cifar10.py --attack_method CW --epsilon 0.0 --set_size 10000 | tee ./logs/NODEFENCE_CIFAR10_CW_0.0.txt
for attack in  JSMA
do
	for epsilon in  0.00784 0.03137 0.06275
	do
		python main_cifar10.py --attack_method $attack --epsilon $epsilon --set_size 10000 | tee ./logs/NODEFENCE_CIFAR10_$attack\_$epsilon.txt
	done
done
