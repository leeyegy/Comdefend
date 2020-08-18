# run in env tf1.14
for attack in FGSM Momentum PGD CW DeepFool 
do
	for epsilon in  0.00784 0.03137 0.06275 
	do
		python compression_imagenet.py --attack_method $attack --epsilon $epsilon --set_size 10000
	done
done
