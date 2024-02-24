# single-GPU training GroupPRCC
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file configs/GroupPRCC/bagtricks_vit.yml  MODEL.DEVICE "cuda:0"
# single-GPU training GroupPRCC
# CUDA_VISIBLE_DEVICES=0,3 python3 tools/train_net.py --config-file configs/GroupPRCC/bagtricks_vit.yml --num-gpus 2