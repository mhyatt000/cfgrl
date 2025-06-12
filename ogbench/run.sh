export MUJOCO_GL=egl
python main.py \
    --env_name=visual-cube-single-play-oraclerep-v0 --train_steps=500000 \
    --agent=agents/cfgrl.py --agent.batch_size=256 \
    --agent.encoder=impala \
    --agent.p_aug=0.5 \
    --eval_interval=1000 --log_interval=1000
