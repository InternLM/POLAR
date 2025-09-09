#!/bin/bash
# Verl PPO training script for Qwen2.5-7B
set -x

# Parameters from original script
nodes=1
train_batch_size=1024
actor_lr=1e-6
critic_lr=1e-5
data_name=MATH
policy_model_name=Qwen2.5-7B-Instruct
reward_model_name=POLAR-7B

# Model paths
actor_path=Qwen/Qwen2.5-7B-Instruct
critic_path=Qwen/Qwen2.5-7B-Instruct

# Data paths
train_data_path=$HOME/data/math/train.parquet
test_data_path=$HOME/data/math/test.parquet

# Reward Configuration
reward_func_path="../src/polar/reward_func.py"

# Experiment name
name="verl_ppo_policy_${policy_model_name}_reward_${reward_model_name}_data_${data_name}"
output_dir="../outputs/${name}"

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Set wandb to offline mode to prevent online sync
# export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

TARGET_FILE="$output_dir/addr_${name}.txt"
RANK=${RANK:-${NODE_RANK:-0}}
MASTER_PORT=6379
MASTER_ADDR=${MASTER_ADDR}
echo "MASTER_ADDR: $MASTER_ADDR"
echo "Rank $RANK is running on $MASTER_ADDR"

if [ "$RANK" -eq 0 ]; then 
    echo "Starting head node (RANK=${RANK}) on port $MASTER_PORT..."
    
    MASTER_ADDR=${MASTER_ADDR}
    echo "$MASTER_ADDR" > "$TARGET_FILE"

    ray start --head --num-gpus 8 --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats --block &
    sleep 30
    
    echo "Executing main program on head node..."

    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0 \
    algorithm.kl_ctrl.type='adaptive' \
    \
    data.train_files="$train_data_path" \
    data.val_files="$test_data_path" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key='prompt' \
    \
    actor_rollout_ref.model.path="$actor_path" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_shm=False \
    \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    \
    critic.model.path="$critic_path" \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.use_remove_padding=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.optim.lr=$critic_lr \
    critic.optim.lr_warmup_steps_ratio=0 \
    critic.optim.warmup_style=cosine \
    critic.optim.min_lr_ratio=0.1 \
    critic.use_dynamic_bsz=False \
    critic.ppo_micro_batch_size_per_gpu=2 \
    \
    reward_model.enable=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=$reward_func_path \
    custom_reward_function.name=compute_score_batch \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nodes \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_ppo_math' \
    trainer.val_before_train=True \
    trainer.experiment_name="$name" \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.default_local_dir=$output_dir \
    \
    trainer.rollout_data_dir="${output_dir}/trajectory_data/rollout" \
    $@

else 
    sleep 10
    MASTER_ADDR=$(cat "$TARGET_FILE")

    echo "Starting worker node (RANK=${RANK}), connecting to ${MASTER_ADDR}:${MASTER_PORT}..."
    ray start --address ${MASTER_ADDR}:${MASTER_PORT}  --num-gpus 8 --block &
    
    sleep 60
    while true; do
        status=$(ray status 2>&1)

        if echo "$status" | grep -q "Active:"; then
            echo "Active nodes found. Sleeping for 10 min..."
            sleep 600
        else
            echo "No active nodes found. Exiting..."
            exit 0
        fi
    done

fi