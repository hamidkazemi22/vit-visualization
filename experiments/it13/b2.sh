echo "Launching 4xjobs : 0"
CUDA_VISIBLE_DEVICES=0 python fore_back.py -n 16 >> run2.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python fore_back.py -n 14 >> run2.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python fore_back.py -n 7 >> run2.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python fore_back.py -n 1 >> run2.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
