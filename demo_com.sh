# run in env tf1.14
for attack in  Momentum FGSM
do
	for epsilon in 0.00784  0.03137 0.06275
	do
		python compression_imagenet.py --attack_method $attack --epsilon $epsilon 
	done
done
