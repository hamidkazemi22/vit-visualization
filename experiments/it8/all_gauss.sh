echo "Launching 4xjobs : 0"
CUDA_VISIBLE_DEVICES=0 python gaus.py -l 0 -g 100. >> g_gauss.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python gaus.py -l 0 -g 10. >> g_gauss.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python gaus.py -l 0 -g 1. >> g_gauss.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python gaus.py -l 0 -g 0.1 >> g_gauss.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 1"
CUDA_VISIBLE_DEVICES=0 python gaus.py -l 0 -g 0.01 >> g_gauss.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python gaus.py -l 2 -g 100. >> g_gauss.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python gaus.py -l 2 -g 10. >> g_gauss.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python gaus.py -l 2 -g 1. >> g_gauss.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 2"
CUDA_VISIBLE_DEVICES=0 python gaus.py -l 2 -g 0.1 >> g_gauss.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python gaus.py -l 2 -g 0.01 >> g_gauss.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python gaus.py -l 4 -g 100. >> g_gauss.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python gaus.py -l 4 -g 10. >> g_gauss.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 3"
CUDA_VISIBLE_DEVICES=0 python gaus.py -l 4 -g 1. >> g_gauss.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python gaus.py -l 4 -g 0.1 >> g_gauss.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python gaus.py -l 4 -g 0.01 >> g_gauss.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python gaus.py -l 6 -g 100. >> g_gauss.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 4"
CUDA_VISIBLE_DEVICES=0 python gaus.py -l 6 -g 10. >> g_gauss.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python gaus.py -l 6 -g 1. >> g_gauss.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python gaus.py -l 6 -g 0.1 >> g_gauss.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python gaus.py -l 6 -g 0.01 >> g_gauss.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 5"
CUDA_VISIBLE_DEVICES=0 python gaus.py -l 8 -g 100. >> g_gauss.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python gaus.py -l 8 -g 10. >> g_gauss.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python gaus.py -l 8 -g 1. >> g_gauss.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python gaus.py -l 8 -g 0.1 >> g_gauss.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 6"
CUDA_VISIBLE_DEVICES=0 python gaus.py -l 8 -g 0.01 >> g_gauss.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python gaus.py -l 10 -g 100. >> g_gauss.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python gaus.py -l 10 -g 10. >> g_gauss.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python gaus.py -l 10 -g 1. >> g_gauss.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 7"
CUDA_VISIBLE_DEVICES=0 python gaus.py -l 10 -g 0.1 >> g_gauss.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python gaus.py -l 10 -g 0.01 >> g_gauss.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 echo "no job" >> g_gauss.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 echo "no job" >> g_gauss.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
