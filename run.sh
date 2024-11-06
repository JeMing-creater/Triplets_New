export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun \
  --nproc_per_node 4 \
  --master_port 29550 \
  Tmain.py