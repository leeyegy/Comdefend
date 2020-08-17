# run in env torch1.1
for attack in PGD
do
	for epsilon in 0.03137
	do
		python ssim.py --attack_method $attack --epsilon $epsilon --set_size 50 | tee ./logs/CIFAR10_ssim_$attack\_$epsilon.txt
	done
done
