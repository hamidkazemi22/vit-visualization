echo "Launching 4xjobs : 0"
CUDA_VISIBLE_DEVICES=0 python high.py -l 2 -g 10. -r 0.01 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 2 -g 1. -r 0.01 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 2 -g 0.1 -r 0.01 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 2 -g 0.01 -r 0.01 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 1"
CUDA_VISIBLE_DEVICES=0 python high.py -l 2 -g 10. -r 0.03 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 2 -g 1. -r 0.03 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 2 -g 0.1 -r 0.03 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 2 -g 0.01 -r 0.03 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 2"
CUDA_VISIBLE_DEVICES=0 python high.py -l 2 -g 10. -r 0.1 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 2 -g 1. -r 0.1 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 2 -g 0.1 -r 0.1 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 2 -g 0.01 -r 0.1 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 3"
CUDA_VISIBLE_DEVICES=0 python high.py -l 2 -g 10. -r 0.3 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 2 -g 1. -r 0.3 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 2 -g 0.1 -r 0.3 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 2 -g 0.01 -r 0.3 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 4"
CUDA_VISIBLE_DEVICES=0 python high.py -l 2 -g 10. -r 1.0 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 2 -g 1. -r 1.0 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 2 -g 0.1 -r 1.0 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 2 -g 0.01 -r 1.0 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 5"
CUDA_VISIBLE_DEVICES=0 python high.py -l 5 -g 10. -r 0.01 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 5 -g 1. -r 0.01 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 5 -g 0.1 -r 0.01 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 5 -g 0.01 -r 0.01 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 6"
CUDA_VISIBLE_DEVICES=0 python high.py -l 5 -g 10. -r 0.03 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 5 -g 1. -r 0.03 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 5 -g 0.1 -r 0.03 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 5 -g 0.01 -r 0.03 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 7"
CUDA_VISIBLE_DEVICES=0 python high.py -l 5 -g 10. -r 0.1 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 5 -g 1. -r 0.1 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 5 -g 0.1 -r 0.1 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 5 -g 0.01 -r 0.1 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 8"
CUDA_VISIBLE_DEVICES=0 python high.py -l 5 -g 10. -r 0.3 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 5 -g 1. -r 0.3 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 5 -g 0.1 -r 0.3 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 5 -g 0.01 -r 0.3 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 9"
CUDA_VISIBLE_DEVICES=0 python high.py -l 5 -g 10. -r 1.0 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 5 -g 1. -r 1.0 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 5 -g 0.1 -r 1.0 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 5 -g 0.01 -r 1.0 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 10"
CUDA_VISIBLE_DEVICES=0 python high.py -l 8 -g 10. -r 0.01 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 8 -g 1. -r 0.01 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 8 -g 0.1 -r 0.01 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 8 -g 0.01 -r 0.01 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 11"
CUDA_VISIBLE_DEVICES=0 python high.py -l 8 -g 10. -r 0.03 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 8 -g 1. -r 0.03 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 8 -g 0.1 -r 0.03 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 8 -g 0.01 -r 0.03 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 12"
CUDA_VISIBLE_DEVICES=0 python high.py -l 8 -g 10. -r 0.1 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 8 -g 1. -r 0.1 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 8 -g 0.1 -r 0.1 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 8 -g 0.01 -r 0.1 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 13"
CUDA_VISIBLE_DEVICES=0 python high.py -l 8 -g 10. -r 0.3 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 8 -g 1. -r 0.3 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 8 -g 0.1 -r 0.3 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 8 -g 0.01 -r 0.3 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 14"
CUDA_VISIBLE_DEVICES=0 python high.py -l 8 -g 10. -r 1.0 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 8 -g 1. -r 1.0 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 8 -g 0.1 -r 1.0 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 8 -g 0.01 -r 1.0 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 15"
CUDA_VISIBLE_DEVICES=0 python high.py -l 11 -g 10. -r 0.01 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 11 -g 1. -r 0.01 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 11 -g 0.1 -r 0.01 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 11 -g 0.01 -r 0.01 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 16"
CUDA_VISIBLE_DEVICES=0 python high.py -l 11 -g 10. -r 0.03 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 11 -g 1. -r 0.03 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 11 -g 0.1 -r 0.03 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 11 -g 0.01 -r 0.03 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 17"
CUDA_VISIBLE_DEVICES=0 python high.py -l 11 -g 10. -r 0.1 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 11 -g 1. -r 0.1 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 11 -g 0.1 -r 0.1 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 11 -g 0.01 -r 0.1 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 18"
CUDA_VISIBLE_DEVICES=0 python high.py -l 11 -g 10. -r 0.3 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 11 -g 1. -r 0.3 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 11 -g 0.1 -r 0.3 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 11 -g 0.01 -r 0.3 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
echo "Launching 4xjobs : 19"
CUDA_VISIBLE_DEVICES=0 python high.py -l 11 -g 10. -r 1.0 >> t_high.sh.txt &
p0=$!
CUDA_VISIBLE_DEVICES=1 python high.py -l 11 -g 1. -r 1.0 >> t_high.sh.txt &
p1=$!
CUDA_VISIBLE_DEVICES=2 python high.py -l 11 -g 0.1 -r 1.0 >> t_high.sh.txt &
p2=$!
CUDA_VISIBLE_DEVICES=3 python high.py -l 11 -g 0.01 -r 1.0 >> t_high.sh.txt &
p3=$!
wait $p0
wait $p1
wait $p2
wait $p3
