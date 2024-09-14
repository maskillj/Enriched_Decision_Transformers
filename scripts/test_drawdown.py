if __name__ == '__main__': 
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    sys.path.append('/Users/jamesmaskill/Desktop/Decision_Transformers/VS_Implementation/src')

    from drawdown_control.decision_transformer import *
    from drawdown_control.decision_transformer_policy import *
    from drawdown_control.trajectory_processing import *
    from drawdown_control.evaluate import *

    import wandb
    from UtilsRL.logger import CompositeLogger
    from torch.utils.data import DataLoader
    from UtilsRL.exp import parse_args, setup
    import d4rl

    args = parse_args("config/walker/walker_medium_replay_v2.py")
    exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
    logger = CompositeLogger(log_dir=f"./log/dt/{args.name}", name=exp_name, logger_config={
        "TensorboardLogger": {}, 
        "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
    })
    setup(args, logger)
    env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward, discard_last=False)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[-1]

    offline_buffer = D4RLTrajectoryBuffer(dataset, seq_len=args.seq_len, return_scale=args.return_scale)
    indices = [i for i, x in enumerate(dataset['terminals']) if x]
    episode_lengths = [indices[0]] + [indices[i] - indices[i-1] for i in range(1, len(indices))]
    max_len = 2048

    dt = DecisionTransformer(
        obs_dim=obs_shape, 
        action_dim=action_shape, 
        embed_dim=args.embed_dim, 
        num_layers=args.num_layers, 
        num_heads=args.num_heads, 
        attention_dropout=args.attention_dropout, 
        residual_dropout=args.residual_dropout, 
        embed_dropout=args.embed_dropout, 
        max_seq_len = max_len
    ).to('cpu')

    policy = DecisionTransformerPolicy(
        dt=dt, 
        state_dim=obs_shape,
        action_dim=action_shape, 
        embed_dim=args.embed_dim, 
        seq_len=args.seq_len, 
        episode_len=args.episode_len, 
        use_abs_timestep=args.use_abs_timestep, 
        policy_type=args.policy_type, 
        device='cpu'
    ).to('cpu')
    policy.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay, betas=args.betas, warmup_steps=args.warmup_steps)

    # main loop   
    policy.train()
    trainloader = DataLoader(
        offline_buffer, 
        batch_size=args.batch_size, 
        pin_memory=True,
        num_workers=args.num_workers
    )
    trainloader_iter = iter(trainloader)
    for i_epoch in range(1, args.max_epoch+1):
        for i_step in range(args.step_per_epoch):
            batch = next(trainloader_iter)
            train_metrics = policy.update(batch, clip_grad=args.clip_grad)
        
        if i_epoch % 2 == 0:
            eval_metrics = eval_decision_transformer(env, policy, [300], args.return_scale, args.eval_episode, seed=args.seed, target_max_drawdown=-3)
            logger.info(f"Episode {i_epoch}: \n{eval_metrics}")
        
        if i_epoch % args.log_interval == 0:
            logger.log_scalars("", train_metrics, step=i_epoch)
            logger.log_scalars("Eval", eval_metrics, step=i_epoch)
            
        if i_epoch % args.save_interval == 0:
            logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/dt/d4rl/{args.name}/{args.task}/seed{args.seed}/policy/")