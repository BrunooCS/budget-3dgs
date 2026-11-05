"""The budget is a hard cap and optimizer state stays aligned with the parameters."""

import torch

from budget3dgs.strategy import BudgetStrategy


def _params(n):
    params = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(torch.randn(n, 3, device='cuda')),
        'scales': torch.nn.Parameter(torch.full((n, 3), -2.0, device='cuda')),
        'quats': torch.nn.Parameter(torch.randn(n, 4, device='cuda')),
        'opacities': torch.nn.Parameter(torch.randn(n, device='cuda') * 3),
        'sh0': torch.nn.Parameter(torch.rand(n, 1, 3, device='cuda'))})
    optimizers = {k: torch.optim.Adam([{'params': p, 'lr': 1e-3, 'name': k}]) for k, p in params.items()}
    return params, optimizers


def test_budget_is_reached_and_never_exceeded():
    torch.manual_seed(0)
    budget = 300
    params, optimizers = _params(200)
    strategy = BudgetStrategy(budget, target='variance', refine_start=0, refine_every=1, recycle=0.01)
    state = strategy.initialize_state(200, 'cuda')
    for step in range(1, 40):
        loss = sum(p.square().sum() for p in params.values())
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()
        n = len(params['means'])
        strategy.accumulate(state, torch.rand(n, 5, device='cuda'))
        strategy.step_post_backward(params, optimizers, state, step, lr=1e-5)
        assert len(params['means']) <= budget
        for k, p in params.items():
            assert optimizers[k].state[p]['exp_avg'].shape == p.shape
        assert state['error'].shape == (len(params['means']),)
    assert len(params['means']) == budget
