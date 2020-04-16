import numpy as np
import tensorflow as tf
import os
import time

from cpprb import ReplayBuffer

from train.trainer import Trainer
from misc.get_relay_buffer import get_replay_buffer, get_default_rb_dict
from misc.utils import save_path, frames_to_gif, is_discrete
from misc.discount_cumsum import discount_cumsum


class OnPolicyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(OnPolicyTrainer, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.replay_buffer = get_replay_buffer(self._policy, self._env)
        kwargs_local_buf = get_default_rb_dict(size=self._policy.horizon, env=self._env)
        kwargs_local_buf["env_dict"]["logp"] = {}
        kwargs_local_buf["env_dict"]["val"] = {}
        if is_discrete(self._env.action_space):
            kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
        self.local_buffer = ReplayBuffer(**kwargs_local_buf)

        episode_steps = 0  # 每次经验轨迹的步数
        episode_return = 0  # 累计折扣奖励
        episode_start_time = time.time()
        total_steps = 0
        n_episode = 0
        obs = self._env.reset()

        tf.summary.experimental.set_step(total_steps)
        while total_steps < self._max_steps:
            for _ in range(self._policy.horizon):
                if self._normalize_obs:

                    obs = np.expand_dims(obs, axis=0)
                    obs = self._obs_normalizer(obs, update=False)
                    obs = np.squeeze(obs, axis=0)

                act, logp, val = self._policy.get_action_and_val(obs)
                next_obs, reward, done, _ = self._env.step(act)
                if self._show_progress:
                    self._env.render()

                episode_steps += 1
                total_steps += 1
                episode_return += reward
                done_flag = done
                if hasattr(self._env, "_max_episode_steps") and episode_steps == self._env._max_episode_steps:
                    done_flag = False
                self.local_buffer.add(
                    obs=obs, act=act, next_obs=next_obs, rew=reward, done=done_flag, logp=logp, val=val
                )
                obs = next_obs

                if done or episode_steps == self._episode_max_steps:
                    tf.summary.experimental.set_step(total_steps)
                    self.finish_horizon()
                    obs =self._env.reset()
                    n_episode += 1
                    fps = episode_steps / (time.time() - episode_start_time)
                    self.logger.info(
                        "Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                            n_episode, int(total_steps), episode_steps, episode_return, fps))
                    tf.summary.scalar(name="Common/training_return", data=episode_return)
                    tf.summary.scalar(name="Common/fps", data=fps)
                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.time()

                # 测试时间间隔
                if total_steps % self._test_interval == 0:
                    avg_test_return = self.evaluate_policy(total_steps)
                    tf.summary.scalar(
                        name="Common/average_test_return", data=avg_test_return)
                    self.writer.flush()

                # 以'_save_model_interval'的时间间隔保存模型参数
                if total_steps % self._save_model_interval == 0:
                    self.checkpoint_manager.save()

            self.finish_horizon(last_val=val)
            tf.summary.experimental.set_step(total_steps)

            # Train actor critic
            if self._policy.normalize_adv:
                samples = self.replay_buffer._encode_sample(np.arange(self._policy.horizon))
                mean_adv = np.mean(samples["adv"])
                std_adv = np.std(samples["adv"])
                if self._normalize_obs:
                    self._obs_normalizer.experience(samples["obs"])

            for _ in range(self._policy.n_epoch):
                samples = self.replay_buffer._encode_sample(np.random.permutation(self._policy.horizon))
                if self._normalize_obs:
                    samples["obs"] = self._obs_normalizer(samples["obs"], update=False)
                if self._policy.normalize_adv:
                    samples["adv"] = (samples["adv"] - mean_adv) / std_adv

                for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                    target = slice(idx * self._policy.batch_size, (idx + 1) * self._policy.batch_size)
                    self._policy.train(
                        states=samples["obs"][target],
                        actions=samples["act"][target],
                        advantages=samples["adv"][target],
                        logp_olds=samples["logp"][target],
                        returns=samples["ret"][target]
                    )

        tf.summary.flush()

    # 计算GAE-Lambda,这个函数当每个轨迹结束或者在epoch终止的时候被调用
    def finish_horizon(self, last_val=0):
        samples = self.local_buffer._encode_sample(np.arange(self.local_buffer.get_stored_size()))
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]   # 时间差分误差集合[δ0, δ1, δ2, ..., δt]
        if self._policy.enable_gae:
            advs = discount_cumsum(
                deltas, self._policy.discount * self._policy.lam
            )
        else:
            advs = deltas

        rets = discount_cumsum(rews, self._policy.discount)[:-1]
        self.replay_buffer.add(
            obs=samples["obs"], act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"])
        )
        self.local_buffer.clear()

    def evaluate_policy(self, total_steps):
        avg_test_return = 0.
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            obs = self._test_env.reset()
            for _ in range(self._episode_max_steps):
                if self._normalize_obs:

                    obs = np.expand_dims(obs, axis=0)
                    obs = self._obs_normalizer(obs, update=False)
                    obs = np.squeeze(obs, axis=0)

                act, _ = self._policy.get_action(obs, test=True)
                act = act if not hasattr(self._env.action_space, "high") else \
                    np.clip(act, self._env.action_space.low, self._env.action_space.high)
                next_obs, reward, done, _ = self._test_env.step(act)
                if self._save_test_path:
                    replay_buffer.add(
                        obs=obs, act=act, next_obs=next_obs,
                        rew=reward, done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))
                elif self._show_test_progress:
                    self._test_env.render()

                episode_return += reward
                obs = next_obs
                if done:
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                total_steps, i, episode_return)
            if self._save_test_path:
                save_path(replay_buffer.sample(self._episode_max_steps),
                          os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return

        return avg_test_return / self._test_episodes