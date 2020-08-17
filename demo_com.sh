# run in env tf1.14
for attack in FGSM
do
	for epsilon in  0.00784 
	do
		python compression_imagenet.py --attack_method $attack --epsilon $epsilon --set_size 100
	done
done
