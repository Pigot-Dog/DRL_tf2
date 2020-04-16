import os
import time
import logging
import argparse

import numpy as np
import tensorflow as tf
from gym.spaces import Box

from misc.utils import save_path, frames_to_gif
from misc.initialize_logger import initialize_logger
from misc.get_relay_buffer import get_replay_buffer
from misc.prepare_output_dir import prepare_output_dir
from misc.normalizer import EmpiricalNormalizer

if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices('GPU'):
        print(cur_device)
        # TODO: （动态）仅在需要时申请显存空间（程序初始运行时消耗很少的显存，随着程序的运行而动态申请显存）
        tf.config.experimental.set_memory_growth(cur_device, enable=True)
        # TODO: （静态）限制消耗固定大小的显存（程序不会超出限定的显存大小，若超出的报错）
        # 单GPU的环境
        # tf.config.experimental.set_virtual_device_configuration(
        #     cur_device,
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        # 单GPU的环境模拟多GPU进行调试
        # tf.config.experimental.set_virtual_device_configuration(
        #     cur_device,
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
        #      tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


class Trainer(object):
    def __init__(self,
                 policy,
                 env, args, test_env=None):
        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env
        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)  # 归一化数据必须为Box类型
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)

        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix)
        )
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir
        )

        # 保存模型
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=self._output_dir, max_to_keep=5
        )
        # 加载模型
        if args.evaluate:
            assert args.model_dir is not None
            self._latest_path_ckpt = tf.train.latest_checkpoint(args.model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()

    def __call__(self):
        pass
        # total_steps = 0
        # tf.summary.experimental.set_step(total_steps)
        # episode_steps = 0
        # episode_return = 0
        # episode_start_time = time.perf_counter()
        # n_episode = 0
        #
        # replay_buffer = get_replay_buffer(
        #     self._policy, self._env, self._use_prioritized_rb,
        #     self._use_nstep_rb, self._n_step
        # )
        #
        # obs = self._env.reset()
        #
        # while total_steps < self._max_steps:
        #     # TODO: ???
        #     if total_steps < self._policy.n_warmup:
        #         action = self._env.action_space.sample()  # 当经验不足n_warmup数量时,采用随机采样
        #     else:
        #         action = self._policy.get_action(obs)
        #
        #     next_obs, reward, done, _ = self._env.step(action)
        #     if self._show_progress:
        #         self._env.render()
        #     episode_steps += 1
        #     episode_return += reward
        #     total_steps += 1
        #     tf.summary.experimental.set_step(total_steps)
        #     done_flag = done
        #     if hasattr(self._env, "_max_episode_steps") and episode_steps == self._env.max_episode_steps:
        #         done_flag = False
        #     replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag)
        #     obs = next_obs
        #
        #     if done or episode_steps == self._episode_max_steps:
        #         obs = self._env.reset()
        #
        #         n_episode += 1
        #         fps = episode_steps / (time.perf_counter() - episode_start_time)
        #         self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
        #             n_episode, total_steps, episode_steps, episode_return, fps
        #         ))
        #         tf.summary.scalar(name="Common/training_return", data=episode_return)
        #
        #         episode_return = 0
        #         episode_steps = 0
        #         episode_start_time = time.perf_counter()
        #
        #     if total_steps < self._policy.n_warmup:
        #         continue
        #
        #     if total_steps % self._policy.update_interval == 0:
        #         samples = replay_buffer.sample(self._policy.batch_size)
        #         with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
        #             self._policy.train(
        #                 samples["obs"], samples["act"], samples["next_obs"],
        #                 samples["rew"], np.array(samples["done"], dtype=np.float32),
        #                 None if not self._use_prioritized_rb else samples["weights"]
        #             )
        #

    def _set_from_args(self, args):
        self._max_steps = args.max_steps
        self._episode_max_steps = args.episode_max_steps \
            if args.episode_max_steps is not None \
            else args.max_steps
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help="Maximum number steps to interact with env")
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help="Maximum steps in an episode")
        parser.add_argument('--n-experiments', type=int, default=1,
                            help="Number of experiments")
        parser.add_argument('--show-progress', action='store_true',
                            help="Call 'render' in training process")
        parser.add_argument('--gpu', type=int, default=0,
                            help="GPU_id")
        parser.add_argument('--save-model-interval', type=int, default=int(1e4),
                            help="Interval to save summary")
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help="Interval to save summary")
        parser.add_argument('--model-dir', type=str, default=None,
                            help="Directory to restore model")
        parser.add_argument('--dir-suffix', type=str, default='',
                            help="Suffix for directory that contains results")
        parser.add_argument('--normalize-obs', action='store_true',
                            help="Normalize observation")
        parser.add_argument('--logdir', type=str, default='results',
                            help="Output directory")
        # test settings
        parser.add_argument('--evaluate', action='store_true',   # 是否要评估模型
                            help='Evaluate trained model')
        parser.add_argument('--test-interval', type=int, default=int(1e4),   # 评估训练模型的时间间隔
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=5,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',  # 保存评价路径
                            help='Save trajectories of evaluation')
        parser.add_argument('--show-test-images', action='store_true',    # action='store_true'表示如果出现，则其值为True，否则为False
                            help='Show input images to neural networks when an episode finishes')
        parser.add_argument('--save-test-movie', action='store_true',   # 保存渲染结果
                            help='Save rendering results')

        # replay buffer 缓冲器
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help="Flag to use prioritized experience replay")
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help="Flag to use nstep experience replay")
        parser.add_argument('--n-step', type=int, default=4,  # 要察看的步骤数
                            help='Number of steps to look over')

        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],  # 日志级别
                            default='INFO', help='Logging level')

        return parser