echo "Launching 4xjobs : 0"
CUDA_VISIBLE_DEVICES=0 python best35.py -g 1 >> best.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python best35.py -g 0 >> best.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python best98.py -g 1 >> best.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python best98.py -g 0 >> best.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
