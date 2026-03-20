import argparse
import os
import pickle
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
# from walk_env_batch_terrain import WalkENV
from Environments.walk_env_batch import WalkENV
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_rsl")
    parser.add_argument("--ckpt", type=int, default=2700)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.device == "cuda" else gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        train_cfg = pickle.load(f)


    env = WalkENV(
        num_envs=1,
        render=True,
        device=args.device,
        t_x=8,
        t_y=20,
        number_of_lanes=1,
        number_of_rows=5
        )


    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    print(f"Loading checkpoint from {resume_path}")
    print(resume_path)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.reset()
    obs = obs.to(args.device)

    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
            # torch.cuda.synchronize()
            # print(done)
            # print()
            
            # obs = obs.to(args.device)
            # if done[0]:
            #     print("Episode ended !!!!.")
            #     obs, _ = env.reset()
            #     obs = obs.to(args.device)

if __name__ == "__main__":
    main()