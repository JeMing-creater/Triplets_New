export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1
torchrun \
  --nproc_per_node 2 \
  --master_port 29555 \
  NewTri.py