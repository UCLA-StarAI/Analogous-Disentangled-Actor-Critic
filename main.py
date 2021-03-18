import numpy as np
import argparse
import sys
import scipy.io as sio
import os
import multiprocessing as mp

sys.path.append("./mems")
sys.path.append("./agents")
sys.path.append("./networks")
sys.path.append("./utils")
sys.path.append("./envs")
sys.path.append("./trainers")

from trainers.Trainer import Trainer
# from managers.RepresentationManager import RepresentationManager

CODE_VERSION = "1.1.0"

def main():
    parser = argparse.ArgumentParser(description = "PyTorch package for RL (by Anji Liu)")
    parser.add_argument("--env-name", type = str, default = "Pendulum-v0",
                        help = "Environment name (default: Pendulum-v0)")
    parser.add_argument("--agent", type = str, default = "DDPG_TD3_AAC",
                        help = "Name of the agent (default: DDPG_TD3_AAC)")
    parser.add_argument("--mode", type = str, default = "train",
                        help = "Running mode from [train, test] (default: train)")
    parser.add_argument("--gamma", type = float, default = 0.99,
                        help = "Discount factor gamma (default: 0.99)")
    parser.add_argument("--seed", type = int, default = 123,
                        help = "Random seed (default: 123)")
    parser.add_argument("--max-buffer-size", type = int, default = 100000,
                        help = "Maximum size of replay memory (default: 100000)")
    parser.add_argument("--training-batch-size", type = int, default = 64,
                        help = "Training batch size (default: 32)")
    parser.add_argument("--max-training-steps", type = int, default = 10000000,
                        help = "Maximum training steps (default: 10000000)")
    parser.add_argument("--max-episode-length", type = int, default = 1000,
                        help = "Maximum length for an episode (default: 1000)")
    parser.add_argument("--warm-up-steps", type = int, default = 200,
                        help = "Steps to warm up the replay memory and others (default: 200)")
    parser.add_argument("--model-saving-interval", type = int, default = 5000,
                        help = "Step interval for saving the model (default: 5000)")
    parser.add_argument("--do-not-save", default = False, action = "store_true",
                        help = "Do not save model (default: False)")
    parser.add_argument("--do-not-load", default = False, action = "store_true",
                        help = "Do not load model (default: False)")
    parser.add_argument("--evaluate-interval", type = int, default = 2000,
                        help = "Evaluation interval (default: 20000)")
    parser.add_argument("--visualize", default = False, action = "store_true",
                        help = "Visualize test run (default: False)")
    parser.add_argument("--evaluate-num-episodes", type = int, default = 20,
                        help = "Number of evaluation episode (default: 20)")
    parser.add_argument("--debug", default = False, action = "store_true",
                        help = "Display debug info (default: False)")
    parser.add_argument("--cuda", type = str, default = "cuda:0",
                        help = "CUDA device (default: cuda:0)")
    parser.add_argument("--max-training-hours", type = float, default = 0,
                        help = "Maximum training hours (default: N/A)")
    parser.add_argument("--cpu", default = False, action = "store_true",
                        help = "Use CPU only (default: False)")
    parser.add_argument("--save-path", type = str, default = "",
                        help = "Save path (default: auto_rule)")

    # For online agents only
    parser.add_argument("--env-num", type = int, default = 16,
                        help = "Number of parallel environments (default: 16)")
    parser.add_argument("--memory-step-length", type = int, default = 8,
                        help = "Number of memory step for each training (default: 8)dr")

    # For Happy Elimination
    parser.add_argument("--level-idx", type = int, default = 1,
                        help = "Level index (default: 1)")
    parser.add_argument("--state-mode", type = int, default = 5,
                        help = "State mode (default: 5)")
    parser.add_argument("--action-mode", type = int, default = 0,
                        help = "Action mode (default: 0)")
    parser.add_argument("--reward-mode", type = int, default = 2,
                        help = "Reward mode (default: 2)")
    parser.add_argument("--terminal-mode", type = int, default = 0,
                        help = "Terminal mode (default: 0)")
    parser.add_argument("--train-multiple-levels", default = False, action = "store_true",
                        help = "Train a single model using multiple levels (default: False)")
    parser.add_argument("--multiple-level-start", type = int, default = 1,
                        help = "Start level of multiple levels (default: 1)")
    parser.add_argument("--multiple-level-end", type = int, default = 50,
                        help = "End level of multiple levels (default: 50)")

    # For SelfDemonstration agents
    parser.add_argument("--max-demo-episodes", type = int, default = 50,
                        help = "Maximum amount of demonstration (default: 50)")

    # Concentration learning
    parser.add_argument("--enable-concentration", default = False, action = "store_true",
                        help = "Enable concentration learning")
    parser.add_argument("--concentration-count", type = int, default = 4,
                        help = "Concentration count (default: 4)")
    parser.add_argument("--cooldown-count", type = int, default = 8,
                        help = "Cool down count (default: 8)")

    # For dataset log display
    parser.add_argument("--train-count", type = int, default = 1,
                        help = "An indicator number of training count (default: 1)")
    parser.add_argument("--eval-num", type = int, default = 100,
                        help = "Evaluation num (default: 100)")
    parser.add_argument("--do-not-post", default = False, action = "store_true",
                        help = "Do not post data to mysql server (default: False)")

    args = parser.parse_args()

    args.code_version = CODE_VERSION
    args.env_extra_dict = {}

    trainer = Trainer(args)
    if args.mode == "train":
        trainer.train()
    else:
        raise NotImplementedError()

    trainer.safe_quit()


if __name__ == "__main__":
    mp.set_start_method("forkserver")

    main()
