echo "Launching 4xjobs : 0"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 0 -f 0 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 1 -f 0 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 2 -f 0 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 3 -f 0 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 1"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 4 -f 0 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 5 -f 0 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 6 -f 0 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 7 -f 0 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 2"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 8 -f 0 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 9 -f 0 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 10 -f 0 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 11 -f 0 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 3"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 0 -f 1 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 1 -f 1 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 2 -f 1 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 3 -f 1 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 4"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 4 -f 1 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 5 -f 1 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 6 -f 1 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 7 -f 1 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 5"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 8 -f 1 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 9 -f 1 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 10 -f 1 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 11 -f 1 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 6"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 0 -f 2 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 1 -f 2 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 2 -f 2 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 3 -f 2 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 7"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 4 -f 2 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 5 -f 2 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 6 -f 2 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 7 -f 2 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 8"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 8 -f 2 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 9 -f 2 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 10 -f 2 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 11 -f 2 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 9"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 0 -f 3 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 1 -f 3 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 2 -f 3 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 3 -f 3 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 10"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 4 -f 3 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 5 -f 3 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 6 -f 3 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 7 -f 3 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 11"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 8 -f 3 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 9 -f 3 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 10 -f 3 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 11 -f 3 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 12"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 0 -f 4 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 1 -f 4 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 2 -f 4 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 3 -f 4 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 13"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 4 -f 4 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 5 -f 4 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 6 -f 4 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 7 -f 4 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 14"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 8 -f 4 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 9 -f 4 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 10 -f 4 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 11 -f 4 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 15"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 0 -f 5 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 1 -f 5 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 2 -f 5 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 3 -f 5 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 16"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 4 -f 5 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 5 -f 5 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 6 -f 5 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 7 -f 5 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 17"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 8 -f 5 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 9 -f 5 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 10 -f 5 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 11 -f 5 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 18"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 0 -f 6 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 1 -f 6 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 2 -f 6 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 3 -f 6 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 19"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 4 -f 6 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 5 -f 6 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 6 -f 6 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 7 -f 6 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 20"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 8 -f 6 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 9 -f 6 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 10 -f 6 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 11 -f 6 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 21"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 0 -f 7 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 1 -f 7 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 2 -f 7 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 3 -f 7 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 22"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 4 -f 7 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 5 -f 7 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 6 -f 7 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 7 -f 7 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 23"
CUDA_VISIBLE_DEVICES=0 python vis.py -l 8 -f 7 >> t_vis.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python vis.py -l 9 -f 7 >> t_vis.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python vis.py -l 10 -f 7 >> t_vis.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python vis.py -l 11 -f 7 >> t_vis.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
