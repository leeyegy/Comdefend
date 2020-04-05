# run in env tf1.14
for attack in  PGD FGSM Momentum 
do
	for epsilon in 0.00784  0.03137 0.06275
	do
		python compression_imagenet.py --attack_method $attack --epsilon $epsilon --set_size 1000
	done
done
for attack in  DeepFool CW STA NONE 
do
	for epsilon in 0.0
	do
		python compression_imagenet.py --attack_method $attack --epsilon $epsilon --set_size 1000
	done
done
