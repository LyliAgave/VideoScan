export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

TASK=videomme,mlvu

CKPT_PATH="./models/my/model"

echo $TASK
echo $CKPT_PATH

cd /gemini/user/private/

python3 -m torch.distributed.run --nproc_per_node='8' -m lmms_eval \
    --model video_scan \
    --model_args pretrained=$CKPT_PATH,conv_template=qwen_1_5,max_frames_num=128,add_kv=8,mm_spatial_pool_mode=average \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_stream \
    --output_path /gemini/user/private/LLaVA-NeXT/eval_logs/ 
    # --verbosity DEBUG
date
