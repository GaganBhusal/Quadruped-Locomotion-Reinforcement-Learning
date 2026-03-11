import os
import shutil
import pickle
import torch
import genesis as gs
import rsl_rl
from rsl_rl.runners import OnPolicyRunner
from walk_env_batch_terrain import WalkENV

def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0005,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 48,
        "save_interval": 10,
        "empirical_normalization": None,
        "seed": 1,
    }

def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    exp_name = "go2_rsl"
    num_envs = 128
    max_iterations = 10000

    env = WalkENV(num_envs=num_envs, render=False, device=gs.device)

    log_dir = f"logs/{exp_name}"
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    # os.makedirs(log_dir, exist_ok=True)

    train_cfg = get_train_cfg(exp_name, max_iterations)
    pickle.dump(train_cfg, open(f"{log_dir}/cfgs.pkl", "wb"))

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    resume_path = "logs/go2_rsl/model_2000.pt"
    print(f"Loading model from: {resume_path}")
    runner.load(resume_path)

    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()