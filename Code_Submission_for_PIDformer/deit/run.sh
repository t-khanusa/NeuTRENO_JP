############## DEIT BASELINE
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 0 --nproc_per_node=4\
 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256\
 --data-path path/to/data --output_dir path/to/dir

############### NEUTRENO DEIT
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 0 --nproc_per_node=4\
 --use_env main.py --model deit_tiny_rlos_former_patch16_224 --batch-size 256\
 --data-path path/to/data --output_dir path/to/dir
