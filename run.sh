export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,7
torchrun \
  --nproc_per_node 4 \
  --master_port 29550 \
  Endomain.py