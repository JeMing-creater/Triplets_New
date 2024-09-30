export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=0
torchrun \
  --nproc_per_node 1 \
  --master_port 29550 \
  main.py