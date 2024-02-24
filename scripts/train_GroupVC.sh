# single-GPU training GroupVC
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file configs/GroupVC/bagtricks_vit.yml  MODEL.DEVICE "cuda:0"

# # multi-GPU training GroupVC
# CUDA_VISIBLE_DEVICES=5,6 python3 tools/train_net.py --config-file configs/GroupVC/bagtricks_vit.yml --num-gpus 2
