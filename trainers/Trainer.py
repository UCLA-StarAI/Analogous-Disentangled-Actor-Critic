import numpy as np
import torch
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import xlrd
from xlutils.copy import copy
import time
import math
import platform

import time
from datetime import datetime
import pytz

from envs.EnvironmentWrapper import EnvironmentWrapper
# from envs.MultipleEnvWrapper import MultipleEnvWrapper
# from envs.MultipleEnvWrapperProc import MultipleEnvWrapperProc
# from envs.AutoTestingMultipleEnv import AutoTestingMultipleEnv

# from utils.dbutil import insert
# from utils.dbutil import calcPassRate

ENABLE_SUPER_FAST_PARALLEL = False

from utils.ColoredPrintFunctions import *

from utils.LogSaver import LogSaver

# from MCTS.MCTSEvaluator import MCTSEvaluator
# from MCTS.ParallelMCTSEvaluator import ParallelMCTSEvaluator

from agents.Agent_DDPG import Agent_DDPG
from agents.Agent_DDPG_TD3_AAC import Agent_DDPG_TD3_AAC
from agents.Agent_DDPG_TD3_AAC_VIME import Agent_DDPG_TD3_AAC_VIME
from agents.Agent_DDPG_TD3_VIME import Agent_DDPG_TD3_VIME
from agents.Agent_DDPG_TD3_AAC_bias_analysis import Agent_DDPG_TD3_AAC_bias_analysis


class Trainer():
    def __init__(self, args):
        # Store parameters
        self.args = args

        # CUDA device
        if args.cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

        # Initialize environment
        if args.env_name == "HappyElimination":
            self.env = EnvironmentWrapper(args.env_name, seed = args.seed, extra_info = args.env_extra_dict)
            self.env_for_eval = EnvironmentWrapper(args.env_name, seed = args.seed, extra_info = args.env_extra_dict)
        else:
            self.env = EnvironmentWrapper(args.env_name, seed = args.seed)
            self.env_for_eval = EnvironmentWrapper(args.env_name, seed = args.seed)

            if hasattr(self.env, "gym_max_episode_steps"):
                self.args.max_episode_length = self.env.gym_max_episode_steps

        # Environment parameters
        self.state_shape = self.env.observation_space
        self.action_type = self.env.action_mode
        self.action_params = dict()
        if self.action_type == "Discrete":
            self.action_params["n"] = self.env.action_n
        elif self.action_type == "Continuous":
            self.action_params["dims"] = self.env.action_dim
            self.action_params["range"] = self.env.action_range

        # Initialize agent
        if args.agent == "DDPG":
            if args.env_name == "HappyElimination":
                self.agent = Agent_DDPGHappyElimOnly(self.state_shape, self.action_type,
                                        self.action_params, args,
                                        device = self.device)
            else:
                self.agent = Agent_DDPG(self.state_shape, self.action_type,
                                    self.action_params, args,
                                    device = self.device)
        elif args.agent == "DDPG_TD3_VIME":
            self.agent = Agent_DDPG_TD3_VIME(self.state_shape, self.action_type,
                                             self.action_params, args,
                                             device = self.device)
        elif args.agent == "DDPG_TD3_AAC":
            self.agent = Agent_DDPG_TD3_AAC(self.state_shape, self.action_type,
                                            self.action_params, args,
                                            device = self.device)
        elif args.agent == "DDPG_TD3_AAC_VIME":
            self.agent = Agent_DDPG_TD3_AAC_VIME(self.state_shape, self.action_type,
                                                 self.action_params, args,
                                                 device = self.device)
        elif args.agent == "DDPG_TD3_AAC_bias_analysis":
            self.agent = Agent_DDPG_TD3_AAC_bias_analysis(self.state_shape,
                    self.action_type, self.action_params, args, device
                    = self.device)
        else:
            raise NotImplementedError()

        self.used_super_fast_parallel = False

        # Stacked environment
        if self.agent.required_training_mode == "on_policy" or \
                self.agent.required_training_mode == "on_policy_double_reward":
            curr_os = platform.platform()
            if len(curr_os) > 6 and curr_os[:6] == "Darwin":
                curr_os = "MacOS"
            elif len(curr_os) > 5 and curr_os[:5] == "Linux":
                curr_os = "Linux"
            else:
                curr_os = "Unknown"

            if args.env_name == "HappyElimination":
                if ENABLE_SUPER_FAST_PARALLEL and curr_os != "MacOS" and \
                        args.env_name == "HappyElimination" and args.mode.find("MCTS") == -1:
                    self.envs = MultipleEnvWrapperProc(
                        args.env_name,
                        args.env_num,
                        seed = args.seed,
                        extra_info = args.env_extra_dict
                    )
                    self.used_super_fast_parallel = True
                else:
                    self.envs = MultipleEnvWrapper(
                        args.env_name,
                        args.env_num,
                        seed = args.seed,
                        extra_info = args.env_extra_dict
                    )
            else:
                self.envs = MultipleEnvWrapper(args.env_name, args.env_num, seed = args.seed)

        # Save Path
        self.save_path = self.construct_save_path()

        # Load model
        if not args.do_not_load:
            if self.agent.load(self.save_path):
                print("> Network loaded")

        # Move agent to GPU, if possible
        self.agent.to()

        # Log saver
        self.logSaver = LogSaver(self.save_path, args)

        # MCTS evaluator
        if self.args.env_name == "HappyElimination":
            self.mctsEvaluator = MCTSEvaluator(self.env, args = args)

        # Enable concentration
        self.enable_concentration = args.enable_concentration
        if self.enable_concentration:
            self.logSaver.enable_platform_sensor()

    def change_agent_to_random(self):
        self.agent = Agent_Random(self.args.level_idx, args = self.args)
        self.args.agent = "Random"

    def train(self):
        if self.agent.required_training_mode == "off_policy":
            self.train_offPolicy()
        elif self.agent.required_training_mode == "on_policy":
            self.train_onPolicy()
        elif self.agent.required_training_mode == "on_policy_double_reward":
            self.train_onPolicy_double_reward()
        else:
            raise NotImplementedError()

    def train_offPolicy(self):
        step = 0
        episode = 0
        episode_step = 0
        episode_reward = 0.0
        state = None

        start_time = time.time()

        correct_episode_reward = 0.0
        support_episode_reward = 0.0

        while step < self.args.max_training_steps:
            # Reset if it is the start of episode
            if state is None:
                state = deepcopy(self.env.reset())
                self.agent.reset()

            # Select action
            action = self.agent.action(state, mode = "train")

            # Interact with environment
            next_state, reward, done, info = self.env.step(action)
            next_state = deepcopy(next_state)

            # For intrinsic reward
            if hasattr(self.agent, "get_augmented_reward_dyna_based"):
                if hasattr(self.agent, "observe_separate_reward") and self.agent.observe_separate_reward:
                    intrinsic_reward = self.agent.get_augmented_reward_dyna_based(state, action, next_state)
                else:
                    reward += self.agent.get_augmented_reward_dyna_based(state, action, next_state)

            # Manually terminate episode if needed
            if self.args.max_episode_length > 0 and episode_step >= self.args.max_episode_length - 1:
                done = True

            # Agent observe environment change
            if hasattr(self.agent, "observe_separate_reward") and self.agent.observe_separate_reward:
                self.agent.observe(state, action, reward, intrinsic_reward, done)
            else:
                self.agent.observe(state, action, reward, done)

            # Train if permitted
            if step >= self.args.warm_up_steps:
                self.agent.train_step()
            
            if hasattr(self.agent, "bias_calculation"):
                self.agent.bias_calculation()

            # Save the model
            if (not self.args.do_not_save) and step % self.args.model_saving_interval == \
                    self.args.model_saving_interval - 1:
                self.agent.save(self.save_path)
                self.logSaver.save_results()

            # Evaluate
            if self.args.evaluate_interval > 0 and step % self.args.evaluate_interval == \
                    self.args.evaluate_interval - 1:
                aveg_reward, _ = self.evaluate()
                prYellow("[Evaluate] #{}: Average episode reward: {}".format(step + 1, aveg_reward))

                end_time = time.time()
                if self.args.max_training_hours != 0 and end_time - start_time > self.args.max_training_hours * 3600:
                    return

                if hasattr(self.agent, "observe_episode_reward"):
                    self.agent.observe_episode_reward(episode_reward)

            # Update status
            step += 1
            episode_step += 1
            episode_reward += reward
            if "correct_reward" in info:
                correct_episode_reward += info["correct_reward"]
                support_episode_reward += info["support_reward"]
            state = next_state

            # Reset if end of an episode
            if done:
                prGreen("#{}: Episode reward: {} steps: {} | correct reward: {} support reward: {}"\
                        .format(episode + 1, episode_reward, step, correct_episode_reward, support_episode_reward))

                # Reset indicators
                state = None
                episode += 1
                episode_step = 0
                episode_reward = 0.0
                correct_episode_reward = 0.0
                support_episode_reward = 0.0

    def train_onPolicy(self, debug_runtime = False):
        step = 0
        curr_step = 0
        episode = 0
        episode_reward = 0.0

        if self.args.max_episode_length > 0:
            self.envs.set_max_episode_length(self.args.max_episode_length)

        start_time = time.time()

        correct_episode_reward = 0.0
        support_episode_reward = 0.0

        states = deepcopy(self.envs.reset())
        while step < self.args.max_training_steps:
            if debug_runtime:
                start_time_s = time.clock()

            # Select actions
            actions = self.agent.action(states, mode = "train")

            # Interact with environment
            next_states, rewards, dones, infos = self.envs.step(actions)
            next_states = deepcopy(next_states)

            if debug_runtime:
                end_time_s = time.clock()

                print("interact", end_time_s - start_time_s)

                start_time_s = time.clock()

            # Agent observe environment change
            self.agent.observe(states, actions, rewards, dones)

            if debug_runtime:
                end_time_s = time.clock()

                print("observe", end_time_s - start_time_s)

                start_time_s = time.clock()

            # Train agent
            self.agent.train_step()

            if debug_runtime:
                end_time_s = time.clock()

                print("train", end_time_s - start_time_s)

            # Save the model
            if (not self.args.do_not_save) and step % self.args.model_saving_interval == \
                    self.args.model_saving_interval - 1:
                self.agent.save(self.save_path)
                self.logSaver.save_results()

            # Evaluate
            if self.args.evaluate_interval > 0 and step % self.args.evaluate_interval == \
                    self.args.evaluate_interval - 1:
                if self.args.env_name == "HappyElimination":
                    aveg_steps, _ = self.evaluate(save_hist = True)
                    prYellow("[Evaluate] #{}: Average step: {}".format(step + 1, aveg_steps))
                else:
                    aveg_reward, _ = self.evaluate()
                    prYellow("[Evaluate] #{}: Average episode reward: {}".format(step + 1, aveg_reward))

                end_time = time.time()
                if self.args.max_training_hours != 0:
                    print("Elapsed time:", (end_time - start_time) / 3600.0)
                if self.args.max_training_hours != 0 and end_time - start_time > self.args.max_training_hours * 3600:
                    return

            # Update status
            step += 1
            episode_reward += rewards[0]
            if "correct_reward" in infos[0]:
                correct_episode_reward += infos[0]["correct_reward"] * math.pow(self.args.gamma, curr_step)
                support_episode_reward += infos[0]["support_reward"] * math.pow(self.args.gamma, curr_step)
                curr_step += 1
            states = next_states

            # Print episode summary
            if dones[0]:
                prGreen("#{0}: Episode reward: {1:.4f} steps: {2} |  ({3:.4f}, {4:.4f})" \
                        .format(episode + 1, episode_reward, step, correct_episode_reward, support_episode_reward))

                episode += 1
                episode_reward = 0.0
                correct_episode_reward = 0.0
                support_episode_reward = 0.0
                curr_step = 0

            if self.enable_concentration:
                if self.logSaver.check_platform():
                    self.envs.enable_concentration_learning(
                        progress = {"total_item_progress": True},
                        count = self.args.concentration_count,
                        cool_down = self.args.cooldown_count
                    )

    def train_onPolicy_double_reward(self):
        assert self.args.reward_mode == 6 or self.args.reward_mode == 7, "Reward mode should be 6 to use A2CSDD"

        step = 0
        curr_step = 0
        episode = 0
        episode_reward = 0.0

        if self.args.max_episode_length > 0:
            self.envs.set_max_episode_length(self.args.max_episode_length)

        start_time = time.time()

        correct_episode_reward = 0.0
        support_episode_reward = 0.0

        states = deepcopy(self.envs.reset())
        while step < self.args.max_training_steps:
            # Select actions
            actions = self.agent.action(states, mode = "train")

            # Interact with environment
            next_states, rewards, dones, infos = self.envs.step(actions)
            correct_rewards = [info["correct_reward"] for info in infos]
            support_rewards = [info["support_reward"] for info in infos]
            next_states = deepcopy(next_states)

            # Agent observe environment change
            self.agent.observe(states, actions, correct_rewards, support_rewards, dones)

            # Train agent
            self.agent.train_step()

            # Save the model
            if (not self.args.do_not_save) and step % self.args.model_saving_interval == \
                    self.args.model_saving_interval - 1:
                self.agent.save(self.save_path)
                self.logSaver.save_results()

            # Evaluate
            if self.args.evaluate_interval > 0 and step % self.args.evaluate_interval == \
                    self.args.evaluate_interval - 1:
                if self.args.env_name == "HappyElimination":
                    aveg_steps, _ = self.evaluate(save_hist = True)
                    prYellow("[Evaluate] #{}: Average step: {}".format(step + 1, aveg_steps))
                else:
                    aveg_reward, _ = self.evaluate()
                    prYellow("[Evaluate] #{}: Average episode reward: {}".format(step + 1, aveg_reward))

                end_time = time.time()
                if self.args.max_training_hours != 0 and end_time - start_time > self.args.max_training_hours * 3600:
                    return

            # Update status
            step += 1
            episode_reward += rewards[0]
            if "correct_reward" in infos[0]:
                correct_episode_reward += infos[0]["correct_reward"] * math.pow(self.args.gamma, curr_step)
                support_episode_reward += infos[0]["support_reward"] * math.pow(self.args.gamma, curr_step)
                curr_step += 1
            states = next_states

            # Print episode summary
            if dones[0]:
                prGreen("#{0}: Episode reward: {1:.4f} steps: {2} |  ({3:.4f}, {4:.4f})" \
                        .format(episode + 1, episode_reward, step, correct_episode_reward, support_episode_reward))

                episode += 1
                episode_reward = 0.0
                correct_episode_reward = 0.0
                support_episode_reward = 0.0
                curr_step = 0

    def test(self, num = 200):
        if self.args.agent == "Random":
            return self.agent.run()
        elif not self.args.train_multiple_levels:
            results = [0 for _ in range(1000)]
            scores = [0 for _ in range(1000)]
            step_counts = []

            if not os.path.exists("save/EvalResult"):
                os.mkdir("save/EvalResult")

            file_path = os.path.join(
                "save/EvalResult/RL",
                self.args.level_version,
                str(self.args.level_idx) + ".txt"
            )

            if not os.path.exists("save/EvalResult/RL"):
                os.mkdir("save/EvalResult/RL")

            if not os.path.exists(os.path.join(
                    "save/EvalResult/RL",
                    self.args.level_version)):
                os.mkdir(os.path.join(
                    "save/EvalResult/RL",
                    self.args.level_version))

            dictionary = dict()
            date_time = datetime.fromtimestamp(int(time.time()),
                                               pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
            dictionary["datetime"] = date_time
            dictionary["train_count"] = self.args.train_count
            dictionary["level_version"] = self.args.level_version
            dictionary["code_version"] = self.args.code_version
            dictionary["hard"] = "RL"
            dictionary["level"] = self.args.level_idx

            for iter in range(num):
                result, score = self.single_step_evaluate()
                result = 999 if result > 999 else result
                results[result] += 1
                scores[result] += score
                step_counts.append(result)
                print("Used {} steps".format(result))

                with open(file_path, "a") as f:
                    f.write(str(result) + " ")

                dictionary["use_step"] = result
                if not self.args.do_not_post:
                    insert(dictionary)

            results = np.array(results)
            scores = np.array(scores)

            calcPassRate(
                dictionary["train_count"],
                dictionary["level"],
                dictionary["level_version"],
                dictionary["code_version"],
                dictionary["hard"]
            )

            '''workbook = copy(xlrd.open_workbook('save/RL.xls'))
            sheet = workbook.get_sheet(0)
            nrows = len(sheet.rows)
            for i, item in enumerate(results):
                sheet.write(i + nrows, 0, self.args.level_idx)
                sheet.write(i + nrows, 1, max(self.env.env.viewParser.moveLeft - int(item), 0))
            workbook.save('save/RL.xls')'''

            # self.logSaver.save_eval_result(results, scores = scores, step_counts = step_counts)

            return results, scores
        else:
            # Train multiple levels
            print("Start evaluating multiple levels")
            level_idxs = range(
                self.args.multiple_level_start,
                self.args.multiple_level_end
            )
            level_means = np.zeros([self.args.multiple_level_end - self.args.multiple_level_start + 1])
            level_stds = np.zeros([self.args.multiple_level_end - self.args.multiple_level_start + 1])

            info = {"level_idx": 0}

            for i, level_idx in enumerate(level_idxs):
                info["level_idx"] = level_idx

                print("Evaluating level {}".format(level_idx))

                results = self.evaluate(info = info)

                level_means[i] = results[0]
                level_stds[i] = results[1]

                print("Evaluated level {}, mean: {}, std: {}".format(level_idx, level_means[i], level_stds[i]))

            return level_idxs, level_means, level_stds

    def test_multi(self, num = 1000):
        if self.args.agent == "Random":
            return self.agent.run()
        elif not self.args.train_multiple_levels:
            final_results = [0 for _ in range(400)]
            final_scores = [0 for _ in range(400)]
            for iter in range(num // self.envs.env_num):
                results, scores = self.single_step_evaluate_multicore()
                for result, score in zip(results, scores):
                    result = 399 if result > 399 else result
                    result = 1 if result < 1 else result
                    final_results[result] += 1
                    final_scores[result] += score
                    print(result)

            results = np.array(final_results)
            scores = np.array(final_scores)

            '''workbook = copy(xlrd.open_workbook('save/RL.xls'))
            sheet = workbook.get_sheet(0)
            nrows = len(sheet.rows)
            for i, item in enumerate(results):
                sheet.write(i + nrows, 0, self.args.level_idx)
                sheet.write(i + nrows, 1, max(self.env.env.viewParser.moveLeft - int(item), 0))
            workbook.save('save/RL.xls')'''

            self.logSaver.save_eval_result(results, scores = scores)

            return results, scores
        else:
            # Train multiple levels
            print("Start evaluating multiple levels")
            level_idxs = range(
                self.args.multiple_level_start,
                self.args.multiple_level_end
            )
            level_means = np.zeros([self.args.multiple_level_end - self.args.multiple_level_start + 1])
            level_stds = np.zeros([self.args.multiple_level_end - self.args.multiple_level_start + 1])

            info = {"level_idx": 0}

            for i, level_idx in enumerate(level_idxs):
                info["level_idx"] = level_idx

                print("Evaluating level {}".format(level_idx))

                results = self.evaluate(info = info)

                level_means[i] = results[0]
                level_stds[i] = results[1]

                print("Evaluated level {}, mean: {}, std: {}".format(level_idx, level_means[i], level_stds[i]))

            return level_idxs, level_means, level_stds

    def snapshot_test(self, info = dict()):
        self.env_for_eval.enable_recording()

        # Reset environment
        state = deepcopy(self.env_for_eval.reset(info = info))
        episode_step = 0
        episode_reward = 0.0

        # Reset agent
        self.agent.reset()

        done = False
        count = 0
        unchanged = False
        while not done:
            # Get action
            action = self.agent.action(state, mode = "test", rand = unchanged)

            # Interact with environment
            next_state, reward, done, info = self.env_for_eval.step(action)
            next_state = deepcopy(next_state)

            if self.args.env_name == "HappyElimination" and info["unchanged"]:
                episode_step -= 1
                unchanged = True
                print("unchanged")
            else:
                unchanged = False

            # Manually terminate episode if needed
            if self.args.max_episode_length > 0 and episode_step >= self.args.max_episode_length - 1:
                done = True

            if count > 800:
                done = True

            # Update
            count += 1
            episode_step += 1
            episode_reward += reward
            state = deepcopy(next_state)

        self.env_for_eval.save_record()

        print(episode_step)

    def snapshot_debug_test(self, info = dict()):
        self.env_for_eval.enable_recording()

        outer_flag = False
        while not outer_flag:
            # Reset agent
            self.agent.reset()

            # Reset environment
            state = deepcopy(self.env_for_eval.reset(info=info))
            episode_step = 0
            episode_reward = 0.0

            done = False
            count = 0
            unchanged = False
            while not done:
                # Get action
                action = self.agent.action(state, mode = "test", rand = unchanged)

                # Interact with environment
                next_state, reward, done, info = self.env_for_eval.step(action)
                next_state = deepcopy(next_state)

                if self.args.env_name == "HappyElimination" and info["unchanged"]:
                    episode_step -= 1
                    unchanged = True
                else:
                    unchanged = False

                # Manually terminate episode if needed
                if self.args.max_episode_length > 0 and episode_step >= self.args.max_episode_length - 1:
                    done = True
                    outer_flag = True

                if count > 800:
                    done = True

                # Update
                count += 1
                episode_step += 1
                episode_reward += reward
                state = deepcopy(next_state)

            print(episode_step)

        self.env_for_eval.save_record()

        print(episode_step)

    def evaluate(self, save_hist = False, get_raw = False, info = dict()):
        results = []
        for episode in range(self.args.evaluate_num_episodes):
            # Reset environment
            state = deepcopy(self.env_for_eval.reset(info = info))
            episode_step = 0
            episode_reward = 0.0

            # Reset agent
            self.agent.reset()

            done = False
            count = 0
            last_unchanged = False
            while not done:
                # Get action
                if last_unchanged:
                    try:
                        action = self.agent.action(state, mode = "test", rand = True)
                    except TypeError:
                        action = self.agent.action(state, mode = "test")
                else:
                    action = self.agent.action(state, mode = "test")

                # Interact with environment
                next_state, reward, done, info = self.env_for_eval.step(action)
                last_unchanged = info["unchanged"] if "unchanged" in info else False
                next_state = deepcopy(next_state)

                if self.args.env_name == "HappyElimination" and info["unchanged"]:
                    episode_step -= 1

                # Manually terminate episode if needed
                if self.args.max_episode_length > 0 and episode_step >= self.args.max_episode_length - 1:
                    done = True

                if count > 1000:
                    done = True

                # Agent observe environment change
                if self.agent.required_training_mode == "off_policy":
                    pass
                    # self.agent.observe(state, action, reward, done)

                # Visualize
                if episode == 0 and self.args.visualize:
                    self.env_for_eval.render()

                # Update
                count += 1
                episode_step += 1
                episode_reward += reward
                state = deepcopy(next_state)

            if self.args.env_name == "HappyElimination":
                results.append(1.0 * episode_step)
            else:
                results.append(1.0 * episode_reward)

        # self.env.reset()

        if not get_raw:
            self.logSaver.record(results)

        if save_hist:
            self.save_histogram(results)

        if not get_raw:
            return np.mean(results), np.std(results)
        else:
            return results

    def single_step_evaluate(self):
        # Reset environment
        state = deepcopy(self.env_for_eval.reset(info = dict()))
        episode_step = 0
        episode_reward = 0.0

        # Reset agent
        self.agent.reset()

        done = False
        count = 0
        last_unchanged = False
        while not done:
            # Get action
            if last_unchanged:
                try:
                    action = self.agent.action(state, mode="test", rand=True)
                except TypeError:
                    action = self.agent.action(state, mode="test")
            else:
                action = self.agent.action(state, mode="test")

            # Interact with environment
            next_state, reward, done, info = self.env_for_eval.step(action)
            last_unchanged = info["unchanged"] if "unchanged" in info else False
            next_state = deepcopy(next_state)

            if self.args.env_name == "HappyElimination" and info["unchanged"]:
                episode_step -= 1

            # Manually terminate episode if needed
            if self.args.max_episode_length > 0 and episode_step >= self.args.max_episode_length - 1:
                done = True

            if count > 1000:
                done = True

            # Update
            count += 1
            episode_step += 1
            episode_reward += reward
            state = deepcopy(next_state)

        if self.args.env_name == "HappyElimination":
            return episode_step, self.env_for_eval.env.get_score()
        else:
            return episode_reward, self.env_for_eval.env.get_score()

    def single_step_evaluate_multicore(self):
        # Reset environment
        states = deepcopy(self.envs.reset())
        episode_steps = [0 for _ in range(self.envs.env_num)]
        episode_rewards = [0.0 for _ in range(self.envs.env_num)]

        # Reset agent
        self.agent.reset()

        final_done = [False for _ in range(self.envs.env_num)]

        done = False
        count = 0
        while not done:
            # Get action
            try:
                actions = self.agent.action(states, mode = "test", rand = True, multi = True)
            except TypeError:
                actions = self.agent.action(states, mode = "test")

            # Interact with environment
            next_states, rewards, dones, infos = self.envs.step(actions, do_not_reset = True)
            next_states = deepcopy(next_states)

            for i in range(self.envs.env_num):
                if dones[i]:
                    final_done[i] = True

            if self.args.env_name == "HappyElimination":
                for i in range(self.envs.env_num):
                    if not final_done[i] and infos[i]["unchanged"]:
                        episode_steps[i] -= 1

            # Manually terminate episode if needed
            if self.args.max_episode_length > 0:
                for i in range(self.envs.env_num):
                    if episode_steps[i] >= self.args.max_episode_length - 1:
                        dones[i] = True

            if count > 1000:
                for i in range(self.envs.env_num):
                    dones[i] = True

            # Update
            count += 1
            for i in range(self.envs.env_num):
                if final_done[i]:
                    continue
                episode_steps[i] += 1
                episode_rewards[i] += rewards[i]
            states = deepcopy(next_states)

            done = True
            for d in dones:
                if not d:
                    done = False

        if self.args.env_name == "HappyElimination":
            return episode_steps, [self.envs.envs[i].env.get_score() for i in range(self.envs.env_num)]
        else:
            return episode_rewards, [self.envs.envs[i].env.get_score() for i in range(self.envs.env_num)]

    def mcts_evaluate(self, evaluation_count = 32, evaluate_mode = "Normal"):
        reward_mode = self.env.env.reward_mode
        self.env.env.set_reward_mode(4)

        self.mctsEvaluator.set_default_policy(lambda state, rand = False: self.agent.action(state, mode = "test", multi = False))
        self.mctsEvaluator.set_prior_prob_func(lambda state: self.agent.action_prob([state])[0])
        results = self.mctsEvaluator.evaluate(evaluate_mode, evaluation_count)

        self.env.env.set_reward_mode(reward_mode)

        '''results = np.reshape(results, (-1,))
        workbook = copy(xlrd.open_workbook('save/MCTS.xls'))
        sheet = workbook.get_sheet(0)
        nrows = len(sheet.rows)

        self.env.env.reset()
        for i, item in enumerate(results):
            sheet.write(i + nrows, 0, self.args.level_idx)
            sheet.write(i + nrows, 1, max(self.env.env.viewParser.moveLeft - int(item), 0))
        workbook.save('save/MCTS.xls')'''

        return np.mean(results), np.std(results)

    def mcts_evaluate_nongreedy(self, evaluation_count = 32):
        reward_mode = self.env.env.reward_mode
        self.env.env.set_reward_mode(4)

        self.mctsEvaluator.set_default_policy(
            lambda state, rand = False: self.agent.action(state, mode = "test", multi = False))
        self.mctsEvaluator.set_prior_prob_func(lambda state: self.agent.action_prob([state])[0])

        results = self.mctsEvaluator.evaluate_normal(evaluation_count)

        self.env.env.set_reward_mode(reward_mode)

        return results

    def mcts_evaluate_record(self, greedy = True):
        reward_mode = self.env.env.reward_mode
        self.env.env.set_reward_mode(4)

        self.mctsEvaluator.set_default_policy(
            lambda state, rand = False: self.agent.action(state, mode = "test", multi = False))
        self.mctsEvaluator.set_prior_prob_func(lambda state: self.agent.action_prob([state])[0])

        if greedy:
            result = self.mctsEvaluator.evaluate_greedy_with_record()
        else:
            result = self.mctsEvaluator.evaluate_nongreedy_with_record()

        self.env.env.set_reward_mode(reward_mode)

    def parallel_mcts_evaluate(self, evaluation_count = 32, env_num = 2):
        reward_mode = self.args.reward_mode
        self.args.reward_mode = 4
        self.args.env_extra_dict["reward_mode"] = 4

        self.save_path = self.construct_save_path()

        agent_args = [self.state_shape, self.action_type,
                      self.action_params, self.args, self.device, self.save_path]

        envs = AutoTestingMultipleEnv(self.args.env_name, env_num, agent_args,
                                      self.args.gamma, self.args.seed,
                                      extra_info = self.args.env_extra_dict)

        cuda = agent_args[3].cuda
        agent_args[3].cuda = "cpu"
        selection_envs = AutoTestingMultipleEnv(self.args.env_name, 4, agent_args,
                                                self.args.gamma, self.args.seed,
                                                extra_info = self.args.env_extra_dict,
                                                need_agent = False)
        agent_args[3].cuda = cuda

        parallelMCTSEvaluator = ParallelMCTSEvaluator(
            env_server = envs,
            selection_env_server = selection_envs,
            default_policy = lambda state, rand = False: self.agent.action(state, mode = "test", multi = False, threshold = 0.9),
            prior_prob_func = lambda state: self.agent.action_prob([state])[0],
            args = self.args
        )

        parallelMCTSEvaluator.evaluate_normal(evaluation_count = evaluation_count)

        parallelMCTSEvaluator.close_envs()

        self.args.reward_mode = reward_mode
        self.args.env_extra_dict["reward_mode"] = reward_mode

    def parallel_mcts_evaluate_count_and_record(self, evaluation_count = 32, env_num = 2):
        reward_mode = self.args.reward_mode
        self.args.reward_mode = 4
        self.args.env_extra_dict["reward_mode"] = 4

        self.save_path = self.construct_save_path()

        agent_args = [self.state_shape, self.action_type,
                      self.action_params, self.args, self.device, self.save_path]

        envs = AutoTestingMultipleEnv(self.args.env_name, env_num, agent_args,
                                      self.args.gamma, self.args.seed,
                                      extra_info = self.args.env_extra_dict)

        cuda = agent_args[3].cuda
        agent_args[3].cuda = "cpu"
        selection_envs = AutoTestingMultipleEnv(self.args.env_name, self.args.expansion_worker_num, agent_args,
                                                self.args.gamma, self.args.seed,
                                                extra_info = self.args.env_extra_dict,
                                                need_agent = False)
        agent_args[3].cuda = cuda

        parallelMCTSEvaluator = ParallelMCTSEvaluator(
            env_server = envs,
            selection_env_server = selection_envs,
            default_policy = lambda state, rand = False: self.agent.action(state, mode = "test", multi = False, threshold = 0.9),
            prior_prob_func = lambda state: self.agent.action_prob([state])[0],
            args = self.args
        )

        parallelMCTSEvaluator.evaluate_normal_with_count_and_record(evaluation_count = evaluation_count)

        parallelMCTSEvaluator.close_envs()

        self.args.reward_mode = reward_mode
        self.args.env_extra_dict["reward_mode"] = reward_mode

    def parallel_mcts_evaluate_record(self, env_num = 2):
        reward_mode = self.args.reward_mode
        self.args.reward_mode = 4
        self.args.env_extra_dict["reward_mode"] = 4

        self.save_path = self.construct_save_path()

        agent_args = [self.state_shape, self.action_type,
                      self.action_params, self.args, self.device, self.save_path]

        envs = AutoTestingMultipleEnv(self.args.env_name, env_num, agent_args,
                                      self.args.gamma, self.args.seed,
                                      extra_info = self.args.env_extra_dict)

        cuda = agent_args[3].cuda
        agent_args[3].cuda = "cpu"
        selection_envs = AutoTestingMultipleEnv(self.args.env_name, 4, agent_args,
                                                self.args.gamma, self.args.seed,
                                                extra_info = self.args.env_extra_dict,
                                                need_agent = False)
        agent_args[3].cuda = cuda

        parallelMCTSEvaluator = ParallelMCTSEvaluator(
            env_server = envs,
            selection_env_server = selection_envs,
            default_policy = lambda state, rand = True: self.agent.action(state, mode = "test", multi = False),
            prior_prob_func = lambda state: self.agent.action_prob([state])[0],
            args = self.args
        )

        parallelMCTSEvaluator.evaluate_normal_with_record()

        parallelMCTSEvaluator.close_envs()

        self.args.reward_mode = reward_mode
        self.args.env_extra_dict["reward_mode"] = reward_mode

    def save_histogram(self, data):
        data = np.reshape(np.array(data), (-1,))
        # print(data.shape)
        plt.hist(data, bins = 10)
        if self.args.env_name == "HappyElimination":
            plt.title(self.args.level_idx)
            plt.savefig(os.path.join(self.save_path, "level_" + str(self.args.level_idx) + ".png"))
        plt.close()

    def construct_save_path(self):
        if self.args.save_path != "":
            return self.args.save_path

        if not os.path.exists("./save"):
            os.mkdir("./save")

        if self.args.agent.find("Atari") != -1:
            folder_path = os.path.join("./save/Atari", self.args.env_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
        elif self.env.env_type == "gym" and self.env.action_mode == "Continuous":
            folder_path = os.path.join("./save/Gym_continuous", self.args.env_name)
            if not os.path.exists("./save/Gym_continuous"):
                os.mkdir("./save/Gym_continuous")
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
        elif self.env.env_type == "myenv":
            folder_path = os.path.join("./save/myenv", self.args.env_name)
            if not os.path.exists("./save/myenv"):
                os.mkdir("./save/myenv")
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
        else:
            folder_path = os.path.join("./save", self.args.env_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

        try:
            agent_name = self.args.agent + "_" + self.agent.mode
        except:
            agent_name = self.args.agent

        folder_path = os.path.join(folder_path, agent_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        if self.args.env_name == "HappyElimination":
            if self.args.train_multiple_levels:
                level_folder = str(self.args.multiple_level_start) + "-" + str(self.args.multiple_level_end)
                folder_path = os.path.join(folder_path, level_folder)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
            else:
                folder_path = os.path.join(folder_path, str(self.args.level_version))
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

                folder_path = os.path.join(folder_path, str(self.args.level_idx))
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

                folder_path = os.path.join(folder_path, "action_mode_" + str(self.args.action_mode))
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

        return folder_path

    def check_value_function(self):
        # Reset environment
        state = deepcopy(self.env_for_eval.reset(info = dict()))

        # Reset agent
        self.agent.reset()

        done = False
        while not done:
            # Get action
            action = self.agent.action(state, mode = "test", rand = True)
            print(action)

            # Interact with environment
            next_state, reward, done, info = self.env_for_eval.step(action)
            next_state = deepcopy(next_state)

            if not info["unchanged"]:
                _, match_score, weight_score = self.agent.get_action_and_match_score(state)
                print(match_score)
                print(weight_score)

                self.env_for_eval.render(render_double = True, agent_hock = self.agent)

                value_gradient = self.agent.get_value_gradient(next_state)
                print("value gradient", value_gradient)
                next_state[1] = next_state[1] * 0.5
                modified_value = self.agent.get_value(next_state)
                print("modified value", modified_value)

            # Update
            state = deepcopy(next_state)

    def safe_quit(self):
        if self.used_super_fast_parallel:
            self.envs.close_envs()
