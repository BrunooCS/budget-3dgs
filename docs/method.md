# How Budget-3DGS chooses where to split

The model holds at most $B$ Gaussians during the whole of training. Every
100 iterations between 500 and 25K, 3DGS-MCMC finds Gaussians with opacity
$\sigma(o_i)\le 0.005$. Each one becomes a copy of a live *parent*, and the split
preserves the rendered opacity and scale. While the model is still below $B$, it
also adds 5% new Gaussians. Parents are sampled with $p_i\propto\sigma(o_i)$.

Budget-3DGS keeps all of this and changes only $p_i$ (`--method rvr`).

## Per-Gaussian residual sums from the backward pass

Let $w_{ip}$ be the blending weight of Gaussian $i$ at pixel $p$ and
$r_p=\hat C_p-C_p$ the RGB render error. The rasterizer renders $K$ extra channels
whose per-Gaussian values $a_i$ are zero, and the loss gains
$\sum_{p,k}A^k_p\,\mathrm{sg}(f^k_p)$. This gives

$$\frac{\partial\mathcal L}{\partial a^k_i}=\sum_p w_{ip}\,f^k_p .$$

Because $a=0$, the extra channels are black and contribute no gradient to any scene
parameter. `tests/test_attribution.py` compares the sums with a brute-force
rendering of every $w_{ip}$ and checks that the training gradients are unchanged.

## Score

With $f=(r,\lVert r\rVert^2,1)$, each view gives $S_i=\sum_p w_{ip}r_p$,
$Q_i=\sum_p w_{ip}\lVert r_p\rVert^2$ and $M_i=\sum_p w_{ip}$. Between two
refinements Budget-3DGS accumulates

$$V_i=\sum_{\text{views}}\Big(Q_i-\frac{\lVert S_i\rVert^2}{M_i}\Big)$$

and samples parents with $p_i\propto V_i$. Within one view a Gaussian adds the same
color to all of its pixels, so it can cancel only the mean error of its footprint.
$V_i$ is what remains, error that changes inside the footprint: edges, texture and
thin structures. Splitting such a Gaussian gives the optimizer two primitives to fit
that detail.

## Ablations (100K Gaussians, 11 scenes, seed 0)

| Parent distribution $p_i\propto$ | PSNR | Δ vs 3DGS-MCMC |
|---|---:|---:|
| $\sigma(o_i)$, gsplat `MCMCStrategy` | 27.38 | — |
| $\sigma(o_i)$, this implementation | 27.39 | +0.01 |
| $\sum w\lvert r\rvert$ (attributed L1 error) | 27.73 | +0.35 |
| $V_i$ (Budget-3DGS) | **27.81** | **+0.44** |
| $V_i$ + reclaiming the 1% least-visible Gaussians | 27.58 | +0.21 |

The opacity-sampled control matches gsplat, so the gain comes from the sampling rule
alone. Matching the Adam-moment handling of the reference implementation is required
for this. Relocation resets the parents' moments, while appended copies start from
zero without resetting their parents. Also resetting parents during growth costs about
0.8 dB on Playroom.

## Limitations

* $V_i$ is an L2 statistic, while training minimizes L1 + D-SSIM. PSNR improves on
  every scene, but LPIPS stays about the same, and Bicycle gets slightly worse.
* Splitting grows detail from Gaussians that are already visible; it cannot create
  geometry in empty regions.
* The extra channels make a training step about 14% slower during the first 25K
  iterations.
