# run in env torch1.1
for attack in  CW DeepFool
do
	for epsilon in  0.00784 0.03137 0.06275
	do
		python main_tiny_imagenet.py --attack_method $attack --epsilon $epsilon --set_size 1000 | tee ./logs/tiny_imagenet/$attack\_$epsilon.txt
	done
done
