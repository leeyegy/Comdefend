# run in env torch1.1
for attack in NONE
do
	for epsilon in 0.0
	do
		python main_cifar10.py --attack_method $attack --epsilon $epsilon --set_size 10000 | tee ./logs/new_CIFAR10_$attack\_$epsilon.txt
	done
done
