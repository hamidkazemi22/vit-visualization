echo "Launching 4xjobs : 0"
CUDA_VISIBLE_DEVICES=0 python fore_back.py -n 34 >> run1.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python fore_back.py -n 35 >> run1.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python fore_back.py -n 36 >> run1.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python fore_back.py -n 37 >> run1.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
