export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1,2,3,4
torchrun \
  --nproc_per_node 4 \
  --master_port 29551 \
  MambaOMain.py