# run in env torch1.1
for attack in  Momentum FGSM
do
	for epsilon in  0.00784 0.03137 0.06275
	do
		python main_cifar10.py --attack_method $attack --epsilon $epsilon | tee ./logs/$attack\_$epsilon.txt
	done
done
