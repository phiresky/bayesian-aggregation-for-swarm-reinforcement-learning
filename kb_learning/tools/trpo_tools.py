import shutil

import numpy as np
import zipfile
import cloudpickle
import tempfile
import os
# from common import logger
import tensorflow as tf


def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob":      obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac":      acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            _, vpred = pi.act(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        # env.render()
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def traj_segment_generator_ma(pi, env, horizon, stochastic, render=False):
    # Initialize state variables
    t = 0
    n_agents = len(env.env.kilobots)
    # ac = np.vstack([env.action_space.sample() for _ in range(n_agents)])
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []
    time_steps = []

    # Initialize history arrays
    sub_sample_thresh = 8
    if n_agents > sub_sample_thresh:
        sub_sample = True
        sub_sample_idx = np.random.choice(n_agents, sub_sample_thresh, replace=False)

        obs = np.array([[ob[ssi] for ssi in sub_sample_idx] for _ in range(horizon)])
        rews = np.zeros([horizon, sub_sample_thresh], 'float32')
        vpreds = np.zeros([horizon, sub_sample_thresh], 'float32')
        news = np.zeros([horizon, sub_sample_thresh], 'int32')
        acs = np.array([ac[sub_sample_idx] for _ in range(horizon)])
        prevacs = acs.copy()
    else:
        sub_sample = False
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros([horizon, n_agents], 'float32')
        vpreds = np.zeros([horizon, n_agents], 'float32')
        news = np.zeros([horizon, n_agents], 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

    while True:
        prevac = ac[sub_sample_idx] if sub_sample else ac
        ac, vpred = pi.act(ob, stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            # yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
            #         "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
            #         "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            if len(ep_rets) == 0:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()

            yield [
                dict(
                    ob=np.array(obs[:, na, :]),
                    rew=np.array(rews[:, na]),
                    vpred=np.array(vpreds[:, na]),
                    new=np.array(news[:, na]),
                    ac=np.array(acs[:, na, :]),
                    prevac=np.array(prevacs[:, na, :]),
                    nextvpred=vpred[na] * (1 - new) if not sub_sample else vpred[sub_sample_idx[na]] * (1 - new),
                    ep_rets=[epr[na] for epr in ep_rets],
                    ep_lens=ep_lens,
                    time_steps=np.array(time_steps)
                ) for na in range(min(n_agents, sub_sample_thresh))
            ]
            _, vpred = pi.act(ob, stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            time_steps = []
        i = t % horizon
        time_steps.append(t)
        obs[i] = [ob[ssi] for ssi in sub_sample_idx] if sub_sample else ob
        vpreds[i] = vpred[sub_sample_idx] if sub_sample else vpred
        news[i] = new
        acs[i] = ac[sub_sample_idx] if sub_sample else ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        if render:
            env.render()
        # rew = np.asarray([rew] * n_agents)
        rews[i] = rew[sub_sample_idx] if sub_sample else rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv_ma(seg, gamma, lam):
    new = [np.append(p["new"], 0) for p in
           seg]  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = [np.append(p["vpred"], p["nextvpred"]) for p in seg]

    for i, p in enumerate(seg):
        T = len(p["rew"])
        p["adv"] = gaelam = np.empty(T, 'float32')
        rew = p["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[i][t + 1]
            delta = rew[t] + gamma * vpred[i][t + 1] * nonterminal - vpred[i][t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        p["tdlamret"] = p["adv"] + p["vpred"]


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.saver = tf.train.Saver()
        self._tmp_dir = None

    def __del__(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    @staticmethod
    def load(path, pol_fn, name='pi', update_params=None):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        if "dim_rec_o" in act_params:
            act_params['ob_space'].dim_rec_o = act_params["dim_rec_o"]
            act_params['ob_space'].dim_local_o = act_params['ob_space'].shape[0] - np.prod(act_params["dim_rec_o"])
            del act_params["dim_rec_o"]

        if update_params:
            act_params.update(update_params)

        act = pol_fn(name=name, **act_params)
        # sess = tf.get_default_session()
        # sess.__enter__()
        aw = ActWrapper(act, act_params)
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            aw.load_state(os.path.join(td, "model"))

        return aw

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def act(self, *args, **kwargs):
        return self._act.act(*args, **kwargs)

    def load_state(self, fname):
        self.saver.restore(tf.get_default_session(), fname)

    def save_state(self, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.saver.save(tf.get_default_session(), fname)

    @property
    def recurrent(self):
        return self._act.recurrent

    @property
    def ob_rms(self):
        if hasattr(self._act, "ob_rms"):
            return self._act.ob_rms
        else:
            return None

    @property
    def ret_rms(self):
        if hasattr(self._act, "ret_rms"):
            return self._act.ret_rms
        else:
            return None

    def save(self, path):
        """Save model to a pickle located at `path`"""
        if os.path.isdir(path):
            path = os.path.join(path, 'model.pkl')

        if self._tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp(prefix='kb_learning')

        # save model parameters
        self.save_state(os.path.join(self._tmp_dir, "model"))

        # create zip archive and write files to archive
        archive_name = os.path.join(self._tmp_dir, "packed.zip")
        with zipfile.ZipFile(archive_name, 'w') as zipf:
            for root, dirs, files in os.walk(self._tmp_dir):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    if file_path != archive_name:
                        zipf.write(file_path, os.path.relpath(file_path, self._tmp_dir))
                        os.remove(file_path)

        # read data from archive
        with open(archive_name, "rb") as f:
            model_data = f.read()
        os.remove(archive_name)

        # dump data to pkl file
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)


