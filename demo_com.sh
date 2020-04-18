# run in env tf1.14
for attack in  NONE
do
	for epsilon in  0.0
	do
		python compression_imagenet.py --attack_method $attack --epsilon $epsilon --set_size 10000
	done
done
