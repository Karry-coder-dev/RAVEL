export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
#export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=en,eth,em,bond
export OMP_NUM_THREADS=2
#export PYTHONUNBUFFERED=1

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=18907 code/main/train_llava_reid.py --config_file=code/main/config/Interactive-PEDES.yaml