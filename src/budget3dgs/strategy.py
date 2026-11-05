"""Budgeted Gaussian relocation driven by residuals attributed through the rasterizer.

The strategy keeps the 3DGS-MCMC machinery (Kheradmand et al., 2024): a hard cap
of ``budget`` primitives, opacity-preserving relocation of dead Gaussians, SGLD
position noise, and 5% growth per refinement until the cap is reached. It changes
the distribution from which relocation and growth parents are drawn:

* ``opacity``   p_i ∝ sigmoid(o_i), exactly 3DGS-MCMC (control).
* ``error``     p_i ∝ E_i = sum_{v,p} w_vip |r_vp|, the attributed L1 residual.
* ``variance``  p_i ∝ V_i = sum_v [ sum_p w_vip ||r_vp||^2 - ||sum_p w_vip r_vp||^2 / sum_p w_vip ],
                the residual energy in a Gaussian's footprint that its own color
                cannot absorb (residual-variance relocation, RVR).

``w_vip`` is the alpha-compositing weight of Gaussian ``i`` at pixel ``p`` of
training view ``v`` and ``r = render - target``. All sums are obtained exactly from
zero-valued probe channels rasterized with the color (see ``train.render``); the
probes leave every other gradient unchanged. Optionally (``recycle > 0``) the
lowest-visible-mass Gaussians are reclaimed as additional donors.
"""

import math
from dataclasses import dataclass, field

import torch
from gsplat.relocation import compute_relocation
from gsplat.strategy.ops import _update_param_with_optimizer, inject_noise_to_position


@dataclass
class BudgetStrategy:
    budget: int
    target: str = 'variance'       # 'variance' (RVR) | 'error' | 'opacity' (3DGS-MCMC)
    recycle: float = 0.0           # budget fraction reclaimed from lowest visible mass per refinement
    noise_lr: float = 5e5
    refine_start: int = 500
    refine_stop: int = 25_000
    refine_every: int = 100
    min_opacity: float = 0.005
    last: dict = field(default_factory=dict, repr=False)  # statistics of the latest refinement, for logging

    @property
    def uses_probes(self):
        return self.target != 'opacity' or self.recycle > 0

    def initialize_state(self, n, device):
        """Running attribution sums and the binomial table used by MCMC relocation."""
        binoms = torch.tensor([[math.comb(n_, k) if k <= n_ else 0 for k in range(51)] for n_ in range(51)],
                              dtype=torch.float32, device=device)
        return {'binoms': binoms, 'error': torch.zeros(n, device=device), 'mass': torch.zeros(n, device=device)}

    @property
    def num_probes(self):
        return 5 if self.target == 'variance' else 2

    def probe_fields(self, residual):
        """Per-pixel fields [H, W, K] attributed through the probes; ``residual`` = render - target, [H, W, 3]."""
        ones = torch.ones_like(residual[..., :1])
        if self.target == 'variance':
            return torch.cat([residual, residual.square().sum(-1, keepdim=True), ones], -1)
        return torch.cat([residual.abs().mean(-1, keepdim=True), ones], -1)

    @torch.no_grad()
    def accumulate(self, state, probe_grad):
        """Add one view's attribution; ``probe_grad`` [N, K] holds the weighted sums of ``probe_fields``."""
        mass = probe_grad[:, -1]
        if self.target == 'variance':
            # weighted residual variance in the footprint: sum w|r|^2 - |sum w r|^2 / sum w
            explained = probe_grad[:, :3].square().sum(-1) / mass.clamp_min(1e-6)
            state['error'] += (probe_grad[:, 3] - explained).clamp_min(0)
        else:
            state['error'] += probe_grad[:, 0]
        state['mass'] += mass

    @torch.no_grad()
    def step_post_backward(self, params, optimizers, state, step, lr):
        """Refine every ``refine_every`` steps inside the refinement window; always inject noise."""
        if self.refine_start < step < self.refine_stop and step % self.refine_every == 0:
            self.last = self._relocate(params, optimizers, state) | {'added': self._grow(params, optimizers, state)}
            state['error'].zero_()
            state['mass'].zero_()
        inject_noise_to_position(params, optimizers, {}, scaler=lr * self.noise_lr)

    # ── Selection rules ──────────────────────────────────

    def _parent_weights(self, params, state):
        opacity = torch.sigmoid(params['opacities'])
        if self.target == 'opacity':
            return opacity
        return state['error'] + 1e-12  # V_i or E_i; the floor keeps unseen Gaussians from making the sum zero

    def _donors(self, params, state):
        opacity = torch.sigmoid(params['opacities'])
        dead = opacity <= self.min_opacity
        if self.recycle > 0:
            k = int(self.recycle * len(opacity))
            mass = state['mass'].masked_fill(dead, float('inf'))
            dead[torch.topk(mass, k, largest=False).indices] = True
        return dead

    # ── Budget-neutral relocation and growth ─────────────

    def _relocate(self, params, optimizers, state):
        dead = self._donors(params, state)
        dead_idx, alive_idx = dead.nonzero(as_tuple=True)[0], (~dead).nonzero(as_tuple=True)[0]
        if len(dead_idx) == 0:
            return {'relocated': 0}
        weights = self._parent_weights(params, state)[alive_idx]
        parents = alive_idx[torch.multinomial(weights, len(dead_idx), replacement=True)]
        self._split(params, optimizers, state, parents, dead_idx)
        return {'relocated': len(dead_idx), 'unique_parents': len(parents.unique()), 'max_copies': int(torch.bincount(parents).max())}

    def _grow(self, params, optimizers, state):
        n = len(params['means'])
        n_new = min(self.budget, int(1.05 * n)) - n
        if n_new <= 0:
            return 0
        parents = torch.multinomial(self._parent_weights(params, state), n_new, replacement=True)
        self._split(params, optimizers, state, parents, None)
        return n_new

    def _split(self, params, optimizers, state, parents, slots):
        """Split ``parents`` with MCMC's opacity/scale-preserving rule into ``slots`` (or append)."""
        eps = torch.finfo(torch.float32).eps
        opacity, scales = torch.sigmoid(params['opacities']), torch.exp(params['scales'])
        new_opacity, new_scales = compute_relocation(opacity[parents], scales[parents],
                                                     torch.bincount(parents)[parents] + 1, state['binoms'])
        new_opacity = new_opacity.clamp(self.min_opacity, 1 - eps)

        def param_fn(name, p):
            if name == 'opacities':
                p[parents] = torch.logit(new_opacity)
            elif name == 'scales':
                p[parents] = torch.log(new_scales)
            if slots is None:
                return torch.nn.Parameter(torch.cat([p, p[parents]]), requires_grad=p.requires_grad)
            p[slots] = p[parents]
            return torch.nn.Parameter(p, requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            # Adam moments follow the 3DGS-MCMC reference: appended copies start at zero, relocation
            # resets only the parents, and reused slots keep their moments.
            if slots is None:
                return torch.cat([v, torch.zeros((len(parents), *v.shape[1:]), device=v.device)])
            v[parents] = 0
            return v

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        if slots is None:
            for key in ('error', 'mass'):
                state[key] = torch.cat([state[key], torch.zeros(len(parents), device=state[key].device)])
