"""GPU (PyTorch/CUDA) engine.

The networks are tiny, so the GPU only pays off when work is batched:
  * GPUPopulation keeps every learner's network, target network, Adam state and
    replay memory stacked in GPU tensors, and trains all of them in ONE batched
    operation (each agent still only learns from its own memory).
  * VecGames plays many games at the same time; every round of every game is a
    single batched forward pass.
Checkpoints use exactly the same files as the CPU version (DQNAgent.save/load),
so you can switch between CPU and GPU at any time.
"""
import numpy as np
import torch
import torch.nn.functional as F

from agent import obs_dim

FIXED_KIND = {"Tit for Tat": 1, "Generous Tit for Tat": 2, "Grim Trigger": 3}
B1, B2, ADAM_EPS, GRAD_CLIP = 0.9, 0.999, 1e-8, 10.0


def _forward(params, x):
    """x: [L, N, in] -> [L, N, 2]; params: [W0 [L,in,h], b0 [L,1,h], ...]"""
    n = len(params) // 2
    h = x
    for i in range(n):
        h = torch.baddbmm(params[2 * i + 1], h, params[2 * i])
        if i < n - 1:
            h = torch.relu(h)
    return h


class GPUPopulation:
    def __init__(self, learners, device):
        self.dev = dev = torch.device(device)
        self.ag = learners
        self.L = L = len(learners)
        c = [a.cfg for a in learners]
        sizes = {a.cfg["batch_size"] for a in learners}
        if len(sizes) > 1:
            raise ValueError("On the GPU all learning agents must use the same batch_size.")
        self.B = sizes.pop()
        t32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
        self.lr = t32([x["lr"] for x in c]).view(L, 1, 1)
        self.gamma = t32([x["gamma"] for x in c]).view(L, 1)
        self.rscale = t32([x.get("reward_scale", 1.0) for x in c])
        self.train_every = np.array([x["train_every"] for x in c])
        self.learn_start = np.array([x["learn_start"] for x in c])
        self.target_every = np.array([x["target_update_every"] for x in c])
        self.cap = np.array([len(a.buf_a) for a in learners])
        self.dim = learners[0].dim

        def stack(get):
            out = []
            for i in range(len(learners[0].net.params)):
                t = t32(np.stack([get(a)[i] for a in learners]))
                out.append(t.unsqueeze(1) if i % 2 else t)  # biases -> [L,1,out]
            return out

        self.p = stack(lambda a: a.net.params)
        for p in self.p:
            p.requires_grad_(True)
        self.tp = stack(lambda a: a.target.params)
        self.m = stack(lambda a: a.net.m)
        self.v = stack(lambda a: a.net.v)
        self.t = t32([a.net.t for a in learners]).view(L, 1, 1)

        CAP = int(self.cap.max())
        self.S = torch.zeros(L, CAP, self.dim, device=dev)
        self.S2 = torch.zeros(L, CAP, self.dim, device=dev)
        self.A = torch.zeros(L, CAP, dtype=torch.long, device=dev)
        self.R = torch.zeros(L, CAP, device=dev)
        self.Dn = torch.zeros(L, CAP, device=dev)
        for l, a in enumerate(learners):
            n = a.buf_size
            if n:
                self.S[l, :n] = t32(a.buf_s[:n])
                self.S2[l, :n] = t32(a.buf_s2[:n])
                self.A[l, :n] = torch.tensor(a.buf_a[:n].astype(np.int64), device=dev)
                self.R[l, :n] = t32(a.buf_r[:n])
                self.Dn[l, :n] = t32(a.buf_d[:n])
        self.size = np.array([a.buf_size for a in learners])
        self.ptr = np.array([a.buf_ptr for a in learners])
        self.pending = np.zeros(L, dtype=np.int64)
        self.ar_L = torch.arange(L, device=dev)
        self.st_loss = torch.zeros(L, device=dev)
        self.st_q = torch.zeros(L, device=dev)
        self.st_n = np.zeros(L, dtype=np.int64)
        for l, a in enumerate(learners):
            a.gpu = (self, l)
        self.use_graph = self.dev.type == "cuda"
        self._graph = None

    # ------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs, owner, eps, greedy):
        P = obs.shape[0]
        q_all = _forward(self.p, obs.unsqueeze(0).expand(self.L, P, self.dim))
        q = q_all[owner, torch.arange(P, device=self.dev)]
        a = q.argmax(1)
        if greedy:
            return a, torch.zeros(P, dtype=torch.bool, device=self.dev)
        explore = torch.rand(P, device=self.dev) < eps
        a = torch.where(explore, torch.randint(0, 2, (P,), device=self.dev), a)
        return a, explore

    @torch.no_grad()
    def q_values(self, l, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.dev).view(1, 1, -1)
        return _forward([p[l:l + 1] for p in self.p], x)[0, 0].cpu().numpy()

    # ----------------------------------------------------------- memory
    @torch.no_grad()
    def store(self, s, a, r, s2, d, players, owners):
        """players: indices into the batch (numpy); owners: learner index of each."""
        if len(players) == 0:
            return
        counts = np.bincount(owners, minlength=self.L)
        order = np.argsort(owners, kind="stable")
        starts = np.cumsum(counts) - counts
        rank = np.empty(len(owners), dtype=np.int64)
        rank[order] = np.arange(len(owners)) - starts[owners[order]]
        pos = (self.ptr[owners] + rank) % self.cap[owners]
        idx = torch.from_numpy(np.stack([players, owners, pos])).to(self.dev)
        pl, ow, ps = idx[0], idx[1], idx[2]
        self.S[ow, ps] = s[pl]
        self.S2[ow, ps] = s2[pl]
        self.A[ow, ps] = a[pl]
        self.R[ow, ps] = r[pl] * self.rscale[ow]
        self.Dn[ow, ps] = d[pl]
        self.ptr = (self.ptr + counts) % self.cap
        self.size = np.minimum(self.size + counts, self.cap)
        self.pending += counts
        for l in np.nonzero(counts)[0]:
            self.ag[l].total_moves += int(counts[l])

    # --------------------------------------------------------- learning
    def train(self):
        ok = self.size >= self.learn_start
        n_upd = np.where(ok, self.pending // self.train_every, 0)
        self.pending = np.where(ok, self.pending - n_upd * self.train_every, self.pending % self.train_every)
        iters = int(n_upd.max()) if len(n_upd) else 0
        if iters == 0:
            return
        masks = torch.tensor(n_upd[None, :] > np.arange(iters)[:, None], dtype=torch.float32, device=self.dev)
        size_t = torch.tensor(self.size, dtype=torch.float32, device=self.dev)
        for it in range(iters):
            mnp = n_upd > it
            self._run_update(masks[it], size_t)
            self.st_n += mnp
            for l in np.nonzero(mnp)[0]:
                ag = self.ag[l]
                ag.total_updates += 1
                if ag.total_updates % self.target_every[l] == 0:
                    with torch.no_grad():
                        for tp, p in zip(self.tp, self.p):
                            tp[l].copy_(p[l])

    def _run_update(self, mask, size_t):
        """Runs _update as a captured CUDA graph (one launch instead of ~100 small ones)."""
        if not self.use_graph:
            return self._update(mask, size_t)
        if self._graph is None:
            self._g_mask = torch.zeros_like(mask)
            self._g_size = size_t.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):  # warm-up with an all-zero mask = no change to any agent
                    self._update(self._g_mask, self._g_size)
            torch.cuda.current_stream().wait_stream(side)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._update(self._g_mask, self._g_size)
        self._g_mask.copy_(mask)
        self._g_size.copy_(size_t)
        self._graph.replay()

    def _update(self, mask, size_t):
        L, B = self.L, self.B
        idx = (torch.rand(L, B, device=self.dev) * size_t[:, None]).long()
        li = self.ar_L[:, None]
        s, a, r, s2, d = self.S[li, idx], self.A[li, idx], self.R[li, idx], self.S2[li, idx], self.Dn[li, idx]
        q_all = _forward(self.p, torch.cat([s, s2], 1))
        with torch.no_grad():
            a2 = q_all[:, B:].argmax(-1, keepdim=True)
            q2 = _forward(self.tp, s2).gather(-1, a2).squeeze(-1)
            y = r + self.gamma * (1.0 - d) * q2
        qa = q_all[:, :B].gather(-1, a.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(qa, y, reduction="none").mean(1)  # Huber, per agent
        grads = torch.autograd.grad((loss * mask).sum(), self.p)
        with torch.no_grad():
            mk = mask.view(L, 1, 1)
            norm = torch.sqrt(sum(g.pow(2).sum(dim=(1, 2)) for g in grads))
            clip = torch.clamp(GRAD_CLIP / (norm + 1e-12), max=1.0).view(L, 1, 1)
            self.t += mk
            tt = self.t.clamp(min=1.0)
            c1 = 1 - torch.pow(B1, tt)
            c2 = 1 - torch.pow(B2, tt)
            for p, g, m, v in zip(self.p, grads, self.m, self.v):
                g = g * clip
                m.add_(mk * ((1 - B1) * (g - m)))
                v.add_(mk * ((1 - B2) * (g * g - v)))
                p.sub_(mk * self.lr * (m / c1) / (torch.sqrt(v / c2) + ADAM_EPS))
            self.st_loss += loss.detach() * mask
            self.st_q += qa.detach().mean(1) * mask

    def flush_stats(self):
        """Move accumulated training stats into each DQNAgent (read per game by the logger)."""
        loss = self.st_loss.cpu().numpy()
        q = self.st_q.cpu().numpy()
        for l, ag in enumerate(self.ag):
            if self.st_n[l]:
                ag.g_loss_sum += float(loss[l])
                ag.g_q_sum += float(q[l])
                ag.g_updates += int(self.st_n[l])
        self.st_loss.zero_()
        self.st_q.zero_()
        self.st_n[:] = 0

    @torch.no_grad()
    def sync_to_agents(self):
        """Copy GPU state back into the DQNAgent objects so they can be saved."""
        get = lambda t, l, i: (t[l, 0] if i % 2 else t[l]).cpu().numpy().astype(np.float32)
        S, S2 = self.S.cpu().numpy(), self.S2.cpu().numpy()
        A, R, Dn = self.A.cpu().numpy(), self.R.cpu().numpy(), self.Dn.cpu().numpy()
        t = self.t.view(-1).cpu().numpy()
        for l, ag in enumerate(self.ag):
            n = len(ag.net.params)
            ag.net.params = [get(self.p[i], l, i) for i in range(n)]
            ag.net.m = [get(self.m[i], l, i) for i in range(n)]
            ag.net.v = [get(self.v[i], l, i) for i in range(n)]
            ag.net.t = int(t[l])
            ag.target.params = [get(self.tp[i], l, i) for i in range(n)]
            c = self.cap[l]
            ag.buf_s[:] = S[l, :c]
            ag.buf_s2[:] = S2[l, :c]
            ag.buf_a[:] = A[l, :c].astype(np.int8)
            ag.buf_r[:] = R[l, :c]
            ag.buf_d[:] = Dn[l, :c]
            ag.buf_ptr, ag.buf_size = int(self.ptr[l]), int(self.size[l])


class VecGames:
    """Many simultaneous games. Player rows 2k and 2k+1 are the two sides of slot k."""

    def __init__(self, pop, agents, n_slots, cfg, device):
        self.pop, self.cfg, self.dev = pop, cfg, torch.device(device)
        self.names = list(agents)
        self.objs = [agents[n] for n in self.names]
        learner_idx = {ag.name: l for l, ag in enumerate(pop.ag)} if pop else {}
        self.owner_of = np.array([learner_idx.get(n, -1) for n in self.names])
        self.kind_of = np.array([0 if self.owner_of[i] >= 0 else FIXED_KIND[n] for i, n in enumerate(self.names)])
        self.K = n_slots
        P = self.P = 2 * n_slots
        self.H = cfg.HISTORY_LEN
        dev = self.dev
        z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
        self.hist = z(P, 4 * self.H)
        self.my_s, self.op_s, self.my_c, self.op_c, self.n_r, self.rew = (z(P) for _ in range(6))
        self.last_opp = torch.full((P,), -1, dtype=torch.long, device=dev)
        self.opp_def = z(P, dtype=torch.bool)
        self.moves = z(P, cfg.MAX_ROUNDS, dtype=torch.int8)
        self.expl = z(P, cfg.MAX_ROUNDS, dtype=torch.bool)
        self.partner = torch.arange(P, device=dev) ^ 1
        self.ar = torch.arange(P, device=dev)
        pay = cfg.PAYOFF
        self.paytab = torch.tensor([pay[(0, 0)], pay[(0, 1)], pay[(1, 0)], pay[(1, 1)]],
                                   dtype=torch.float32, device=dev)
        self.agent_of = np.zeros(P, dtype=np.int64)
        self.owner_t = torch.zeros(P, dtype=torch.long, device=dev)
        self.kind_t = torch.zeros(P, dtype=torch.long, device=dev)
        self.rounds = np.zeros(n_slots, dtype=np.int64)
        self.length = np.zeros(n_slots, dtype=np.int64)
        self.active = np.zeros(n_slots, dtype=bool)
        self.eps0 = np.zeros(P)
        self.bonus = any((cfg.WIN_BONUS, cfg.TIE_PENALTY, cfg.LOSS_PENALTY))

    def start(self, slots, a1, a2, lengths, eps_by_agent):
        slots = np.asarray(slots)
        rows = np.concatenate([2 * slots, 2 * slots + 1])
        self.agent_of[2 * slots] = a1
        self.agent_of[2 * slots + 1] = a2
        self.rounds[slots] = 0
        self.length[slots] = lengths
        self.active[slots] = True
        self.eps0[rows] = eps_by_agent[self.agent_of[rows]]
        r = torch.from_numpy(rows).to(self.dev)
        for t in (self.hist, self.my_s, self.op_s, self.my_c, self.op_c, self.n_r, self.rew, self.opp_def):
            t[r] = 0
        self.last_opp[r] = -1
        own = np.maximum(self.owner_of[self.agent_of], 0)
        self.owner_t = torch.from_numpy(own).to(self.dev)
        self.kind_t = torch.from_numpy(self.kind_of[self.agent_of]).to(self.dev)

    def obs(self):
        n = self.n_r.clamp(min=1)
        has = self.n_r > 0
        return torch.cat([
            self.hist,
            torch.tanh((self.my_s - self.op_s) / 15.0)[:, None],
            torch.where(has, self.my_c / n, 0.5)[:, None],
            torch.where(has, self.op_c / n, 0.5)[:, None]], 1)

    def step(self, greedy, learn, eps_by_agent=None):
        """Plays one round in every active slot. Returns the slots whose game just ended."""
        dev = self.dev
        s = self.obs()
        if self.pop is not None:
            eps = torch.from_numpy(np.where(self.owner_of[self.agent_of] >= 0,
                                            eps_by_agent[self.agent_of], 0.0).astype(np.float32)).to(dev)
            a, explore = self.pop.act(s, self.owner_t, eps, greedy)
        else:
            a = torch.zeros(self.P, dtype=torch.long, device=dev)
            explore = torch.zeros(self.P, dtype=torch.bool, device=dev)
        # fixed strategies
        k = self.kind_t
        tft = torch.where(self.last_opp < 0, 0, self.last_opp)
        forgive = (self.last_opp == 1) & (torch.rand(self.P, device=dev) < self.cfg.GENEROUS_TFT_FORGIVENESS)
        gtft = torch.where(forgive, 0, tft)
        grim = self.opp_def.long()
        a = torch.where(k == 1, tft, torch.where(k == 2, gtft, torch.where(k == 3, grim, a)))
        explore = explore & (k == 0)

        o = a[self.partner]
        # finished games (e.g. in a tournament, waiting for the longest game) must not change any more
        live = torch.from_numpy(np.repeat(self.active, 2).astype(np.float32)).to(dev)
        pay = self.paytab[a * 2 + o] * live
        ri = self.n_r.long().clamp(max=self.cfg.MAX_ROUNDS - 1)
        self.moves[self.ar, ri] = a.to(torch.int8)
        self.expl[self.ar, ri] = explore
        self.my_s += pay
        self.op_s += pay[self.partner]
        self.my_c += (a == 0) * live
        self.op_c += (o == 0) * live
        self.n_r += live
        af, of = a.float(), o.float()
        self.hist = torch.cat([torch.stack([1 - af, af, 1 - of, of], 1), self.hist[:, :-4]], 1)
        self.last_opp = o
        self.opp_def |= (o == 1)

        self.rounds[self.active] += 1
        done_slots = self.active & (self.rounds >= self.length)
        done_p = np.repeat(done_slots, 2)
        done_t = torch.from_numpy(done_p.astype(np.float32)).to(dev)
        reward = pay
        if self.bonus and done_slots.any():
            c = self.cfg
            b = torch.where(self.my_s > self.op_s, c.WIN_BONUS,
                            torch.where(self.my_s == self.op_s, -c.TIE_PENALTY, -c.LOSS_PENALTY))
            reward = reward + b * done_t
        self.rew += reward

        if learn:
            s2 = self.obs()
            players = np.nonzero(np.repeat(self.active, 2) & (self.owner_of[self.agent_of] >= 0))[0]
            self.pop.store(s, a, reward, s2, done_t, players, self.owner_of[self.agent_of[players]])
            self.pop.train()
        self.active &= ~done_slots
        return np.nonzero(done_slots)[0]

    def collect(self, slots):
        """Finished games as plain Python data (same shape as play_training_game's result)."""
        rows = torch.from_numpy(np.concatenate([2 * slots, 2 * slots + 1])).to(self.dev)
        maxlen = int(self.length[slots].max())
        mv = self.moves[rows, :maxlen].cpu().numpy()
        ex = self.expl[rows, :maxlen].cpu().numpy()
        sc = self.my_s[rows].cpu().numpy()
        rw = self.rew[rows].cpu().numpy()
        n = len(slots)
        out = []
        for j, sl in enumerate(slots):
            L = int(self.length[sl])
            ma, mb = mv[j, :L].tolist(), mv[n + j, :L].tolist()
            out.append(dict(
                n1=self.names[self.agent_of[2 * sl]], n2=self.names[self.agent_of[2 * sl + 1]],
                ha=list(zip(ma, mb)), hb=list(zip(mb, ma)),
                xa=ex[j, :L].tolist(), xb=ex[n + j, :L].tolist(),
                sa=int(sc[j]), sb=int(sc[n + j]), ra=float(rw[j]), rb=float(rw[n + j]), rounds=L,
                eps_a=float(self.eps0[2 * sl]), eps_b=float(self.eps0[2 * sl + 1])))
        return out
