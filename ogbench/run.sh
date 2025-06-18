export MUJOCO_GL=egl
uv run main.py \
    --env_name=cube-single-play-oraclerep-v0 --train_steps=500000 \
    --agent.batch_size=4096 \
    --agent.p_aug=0.5 \
    --eval_interval=5000 --log_interval=1000 \
    --video_episodes=2 \
    # --agent=agents/cfgrl.py
    # --agent.encoder=impala \
