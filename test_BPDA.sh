for iter in 5 10 15
do
	for epsilon in 0.03137 0.06275
	do 
		python test_BPDA.py --max_iterations $iter --epsilon $epsilon | tee logs/BPDA-$iter\-$epsilon.txti
	done
done
