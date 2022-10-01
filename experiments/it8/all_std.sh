echo "Launching 4xjobs : 0"
CUDA_VISIBLE_DEVICES=0 python cj_std.py -l 0 -g 10. >> a_std.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python cj_std.py -l 0 -g 1. >> a_std.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python cj_std.py -l 0 -g 0.1 >> a_std.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python cj_std.py -l 0 -g 0.01 >> a_std.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 1"
CUDA_VISIBLE_DEVICES=0 python cj_std.py -l 2 -g 10. >> a_std.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python cj_std.py -l 2 -g 1. >> a_std.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python cj_std.py -l 2 -g 0.1 >> a_std.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python cj_std.py -l 2 -g 0.01 >> a_std.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 2"
CUDA_VISIBLE_DEVICES=0 python cj_std.py -l 4 -g 10. >> a_std.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python cj_std.py -l 4 -g 1. >> a_std.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python cj_std.py -l 4 -g 0.1 >> a_std.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python cj_std.py -l 4 -g 0.01 >> a_std.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 3"
CUDA_VISIBLE_DEVICES=0 python cj_std.py -l 6 -g 10. >> a_std.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python cj_std.py -l 6 -g 1. >> a_std.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python cj_std.py -l 6 -g 0.1 >> a_std.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python cj_std.py -l 6 -g 0.01 >> a_std.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 4"
CUDA_VISIBLE_DEVICES=0 python cj_std.py -l 8 -g 10. >> a_std.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python cj_std.py -l 8 -g 1. >> a_std.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python cj_std.py -l 8 -g 0.1 >> a_std.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python cj_std.py -l 8 -g 0.01 >> a_std.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 5"
CUDA_VISIBLE_DEVICES=0 python cj_std.py -l 10 -g 10. >> a_std.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python cj_std.py -l 10 -g 1. >> a_std.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python cj_std.py -l 10 -g 0.1 >> a_std.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python cj_std.py -l 10 -g 0.01 >> a_std.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
