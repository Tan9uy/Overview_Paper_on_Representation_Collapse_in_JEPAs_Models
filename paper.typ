#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/fletcher:0.5.5": diagram, edge, node
#import "@preview/cetz:0.3.2"

#show: ieee.with(
  title: [The Challenge of Representation Collapse in JEPAs Models],
  abstract: [
    Joint Embedding Predictive Architectures (JEPAs) represent a significant advancement in self-supervised learning. Unlike traditional generative approaches that reconstruct inputs in the original modality (e.g., pixels), JEPAs predict representations directly in latent space. While JEPAs offer high sample efficiency and robustness compared to generative methods, they introduce a critical vulnerability: representation collapse. Preventing this failure mode is essential to ensure that learned embeddings remain informative and preserve fine-grained features for downstream tasks. This overview describes the JEPA framework, provides mathematical definitions of collapse, and outlines mitigation paradigms—including historical contrastive approaches, architectural asymmetry, masking strategies, explicit regularization, and state-of-the-art methods like SIGReg. Finally, we discuss known limitations discussed in the literature.
  ],
  authors: (
    (
      name: "Pierre-Emmanuel Baumier, Tanguy Paymal",
      department: [Pharmacy Faculty],
      organization: [UGA],
      location: [Grenoble, France],
      email: "tanguy.paymal@etu.univ-grenoble-alpes.fr",
    ),
  ),
  figure-supplement: [Fig.],
)

= Introduction


Deep learning in visual applications like radiology or pathology relies heavily on standard representations, which traditionally required large volumes of annotated data. Self-supervised learning (SSL) methods avoid this bottleneck by pretraining models on unlabeled data. Recently, Joint Embedding Predictive Architectures (JEPAs), introduced by LeCun @lecun2022path, have been recognized as a powerful SSL framework. Unlike generative models that reconstruct raw pixels or signals, JEPAs perform prediction directly within an abstract representation space @assran2023ijepa.

However, predicting in latent space introduces a fundamental failure mode: *representation collapse*, where the encoder maps distinct inputs to near-constant or low-dimensional vectors. While complete collapse manifests through trivial constant outputs, models also face a more insidious threat: *partial* or *dimensional collapse*. During a partial collapse, training loss metrics may remain stable while crucial semantic features are quietly erased because they are mapped to a smaller subspace. This overview provides a high-level summary of the JEPA framework, its collapse mechanisms, and state-of-the-art mitigation techniques alongside their inherent limitations.



= Theoretical Architecture of JEPA (LeCun Framework)

Introduced by Yann LeCun @lecun2022path, the Joint-Embedding Predictive Architecture (JEPA), described in @fig-lecun-jepa, replaces generative pixel-space autoencoding with predictions in representation space. The core architecture consists of two encoders, a context encoder $E_(theta_x)$ and a target encoder $E_(theta_y)$, and a predictor $P_phi$ conditioned on a latent variable $z$, where $x$ and $y$ are two views or spatio-temporal segments of the same input signal (e.g., unmasked context and masked target regions), and $z$ represents latent positional or structural variables.

#figure(
  cetz.canvas({
    import cetz.draw: *
    set-style(
      rect: (radius: 0.08, stroke: 0.7pt),
      content: (padding: 2.0pt),
    )

    // Input Nodes
    rect((-1.7, 0.0), (1.3, 0.6), name: "x", fill: rgb("#EBF3FA"))
    content("x", text(size: 8pt, weight: "medium")[Context $x$])

    rect((3.7, 0.0), (6.7, 0.6), name: "y", fill: rgb("#EBF3FA"))
    content("y", text(size: 8pt, weight: "medium")[Target $y$])

    // Encoder Nodes
    rect((-1.7, 1.5), (1.3, 2.5), name: "enc_x", fill: rgb("#FFFFFF"))
    content("enc_x", text(size: 8pt)[
      *Context Encoder* \
      $E_(theta_x)(x) = s_x$
    ])

    rect((3.7, 1.5), (6.7, 2.5), name: "enc_y", fill: rgb("#FFFFFF"))
    content("enc_y", text(size: 8pt)[
      *Target Encoder* \
      $E_(theta_y)(y) = hat(s)_y$
    ])

    // Predictor Node
    rect((-1.7, 3.4), (1.3, 4.5), name: "pred", fill: rgb("#FFFFFF"))
    content("pred", text(size: 8pt)[
      *Predictor Network* \
      $P_phi (s_x, z) = hat(s)_x$
    ])

    // Spatial Latent Variable z
    rect((2.6, 3.0), (3.6, 3.6), name: "z", fill: rgb("#FEF5E7"))
    content("z", text(size: 7.5pt)[Latent $z$])

    // Loss Objective Node
    rect(
      (0.0, 5.4),
      (5.0, 6.3),
      name: "loss",
      fill: rgb("#FDEDEC"),
      stroke: 0.8pt + rgb("#C0392B"),
    )
    content("loss", text(
      size: 8.5pt,
      weight: "bold",
      fill: rgb("#922B21"),
    )[
      Energy / Prediction Criterion \
      $cal(E)(x, y) = D(P_phi (E_(theta_x)(x), z), E_(theta_y)(y))$
    ])

    // Connections
    line("x.north", "enc_x.south", mark: (end: ">", size: 0.15))
    line("y.north", "enc_y.south", mark: (end: ">", size: 0.15))

    line("enc_x.north", "pred.south", mark: (end: ">", size: 0.15))
    line("z.west", "pred.east", mark: (end: ">", size: 0.15))

    line("pred.north", (-0.2, 4.95), (1.2, 4.95), "loss.south", mark: (
      end: ">",
      size: 0.15,
    ))

    line("enc_y.north", (5.2, 4.95), (3.8, 4.95), "loss.south", mark: (
      end: ">",
      size: 0.15,
    ))
  }),
  caption: [Theoretical schematic of the Joint-Embedding Predictive Architecture (JEPA) as conceptualized by LeCun @lecun2022path.],
) <fig-lecun-jepa>

The main objective is to minimize the energy distance metric $D$:

$
  cal(E)(x,y) = D(hat(s)_y, s_y) = norm(P_phi (E_(theta_x)(x), z) - E_(theta_y)(y))^2
$

To prevent representation collapse, LeCun proposes to *Maximize the Information Content* in both encoders.
The system must maximize mutual information $I(X; s_x)$ and $I(Y; s_y)$, or maximize latent feature covariance volume:
$
  max_(theta) "Vol"( "Cov"(s) ) quad upright("or") quad max_(theta) I(X; E_(theta)(X))
$

If latent variance is unconstrained, naive end-to-end gradient minimization causes both encoders to collapse into a constant output.

= Representation Collapse: Definitions and Mechanisms

In self-supervised learning, representation collapse occurs when embedding functions map distinct inputs into a low-rank or constant subspace @jing2022dimensional.

Consider a batch embedding matrix $Z in RR^(N times d)$ representing target embeddings ($Z_i = E_theta (y_i)^T$) across $N$ samples in $d$ embedding dimensions:

$
  Z = mat(
    E_theta (y_1)^T;
    E_theta (y_2)^T;
    dots.v;
    E_theta (y_N)^T
  ) in RR^(N times d)
$

Assuming zero-centered embeddings ($1/N sum_i Z_(i, :) = 0$), the empirical covariance matrix $C in RR^(d times d)$ is:

$ C = 1 / N Z^T Z $

Let $sigma_1 >= sigma_2 >= ... >= sigma_d >= 0$ be the singular values of $Z$.

== Complete Collapse
Complete collapse occurs when encoders degenerate to a constant map ($E_(theta_x)(x) = E_(theta_y)(y) = c, thin forall x, y$).
$
  "Var"(Z_((:, j))) = 0, quad forall j in {1, ..., d} \
  => D(P_phi (E_(theta_x)(x), z), E_(theta_y)(y)) = 0
$

In this mode, the loss is minimized because the model has learned to ignore the input information, discarding all semantics.

== Dimensional Collapse
In dimensional collapse, output variance remains positive ($"Var"(Z_((:, j))) > 0$), but embedding vectors are restricted to a low-rank subspace $k << d$ @jing2022dimensional. The singular values converge rapidly and exponentially toward zero ($sigma_k -> 0$). To see this failure mode, the **Effective Rank** ($"ER"(C)$) can be computed:

$
  "ER"(C) = exp(- sum_(i=1)^d p_i log p_i), quad p_i = (sigma_i) / (sum_(j=1)^d sigma_j)
$

When $"ER"(C) << d$, orthogonal feature directions are suppressed, erasing subtle information.


= Methods to Prevent Representation Collapse

Preventing representation collapse in joint-embedding architectures (JEA) relies on distinct algorithmic and data strategies:

== Contrastive Methods and Dual-Term Objectives

Historically, contrastive learning methods (e.g., SimCLR @chen2020simclr) prevented collapse in joint embeddings by using objectives like InfoNCE @oord2018infonce. These objectives pull positive pair embeddings together while explicitly repelling negative pairs. While standard JEPA models avoid contrastive pairs due to their exponential computational cost @lecun2022path, which grows with the dimension of the embeddings, some recent hybrid JEPA @cjepa extensions reintroduce secondary contrastive loss terms to enforce global discriminability alongside local representation prediction.

== Architectural Asymmetry and Capacity Bottlenecks
Non-contrastive models introduce structural or dynamic asymmetries between processing branches to avoid identity shortcuts:

1. *Input Views and Masking Strategies*: Feeding different context inputs ($x$) versus target inputs ($y$) forces invariant predictions. In image JEPAs (like I-JEPA @assran2023ijepa), multi-block masking is crucial: if masked regions are too small or localized, the model easily learns trivial local pixel interpolation rather than semantic predictions, leading to partial collapse.
2. *Asymmetric Predictor Head and Information Bottleneck*: Placing a predictor $P_phi$ strictly on the context branch breaks architectural symmetry. Furthermore, constraining $P_phi$ to have significantly lower capacity (e.g., fewer Transformer layers) than the encoders acts as an information bottleneck. The context encoder is forced to produce highly expressive features so that the weak predictor can still accurately infer target embeddings.
3. *Stop-Gradient Operator ($"sg"[dot]$)*: Blocking backpropagation along the target encoder pathway breaks the joint optimization loop:
  $ cal(L) = D(P_phi (E_(theta_x)(x), z), "sg"[E_(theta_y)(y)]) $
  Gradients propagate only through $theta_x$ and $phi$, setting $nabla_(theta_y) cal(L) := 0$.
4. *Exponential Moving Average (EMA)*: Introduced in SSL by BYOL @grill2020byol and later used in I-JEPA @assran2023ijepa, the target encoder parameters $theta_y$ update via a momentum combination with the weights from the encoder $theta_x$:
  $ theta_y^((t)) <- tau theta_y^((t-1)) + (1 - tau) theta_x^((t)) $
  It is not fully understood why methods like this one can stabilize training and avoid collapse. For example, Chen & He @chen2021simsiam argue that it is not strictly necessary and Stop-Gradient should be enough.

== Information Maximization and Variance-Covariance Regularization (VICReg)

VICReg @bardes2022vicreg avoids collapse by optimizing a loss with three distinct terms: an **Invariance criterion** $s(Z_x, Z_y)$, and explicit **Variance** $v(Z)$ and **Covariance** $c(Z)$ regularization penalties applied to both embedding sets $Z_x, Z_y in RR^(N times d)$:

$
  cal(L) & = lambda s(Z_x, Z_y) + mu [v(Z_x) + v(Z_y)] \
         & quad + nu [c(Z_x) + c(Z_y)]
$

where $lambda, mu, nu > 0$ are hyperparameter weighting coefficients.

1. *Invariance Term $s(Z_x, Z_y)$*: Minimizes the mean-squared Euclidean distance between corresponding context/target embedding vectors $z_(x, i)$ and $z_(y, i)$ across the batch $N$, without applying $L_2$ normalization :
  $ s(Z_x, Z_y) = 1 / N sum_(i=1)^N norm(z_(x, i) - z_(y, i))_2^2 $

2. *Variance Term $v(Z)$*: Prevents complete collapse by applying a threshold $gamma > 0$ (typically $gamma = 1$) on the standard deviation along each feature dimension $j in {1, ..., d}$ of a batch :
  $ v(Z) = 1 / d sum_(j=1)^d max(0, gamma - sqrt("Var"(Z_((:, j))) + epsilon)) $
  where $epsilon > 0$ is a small scalar preventing numerical instability near zero derivative.

3. *Covariance Term $c(Z)$*: Prevents dimensional collapse by decorrelating distinct feature dimensions $i != j$. The off-diagonal entries of the empirical covariance matrix $C = 1 / (N-1) Z^T Z$ are pushed toward zero:
  $ c(Z) = 1 / d sum_(i != j) (C_(i, j))^2 $

== Sketched Isotropic Gaussian Regularization (SIGReg)

To prevent representation collapse, Balestriero & LeCun introduced SIGReg (@balestriero2025sigreg, @lewm2026), proving that forcing latent representations into an isotropic Gaussian $cal(N)(0, I)$ minimizes expected prediction risk across downstream tasks.

By the *Cramér-Wold theorem*, $Z$ is isotropic Gaussian if and only if every 1D projection $Z u$ along a unit direction $u in SS^(d-1)$ follows $cal(N)(0, 1)$. SIGReg operationalizes this by projecting $Z in RR^(N times d)$ along $M$ random unit directions $U in RR^(d times M)$ and penalizing non-Gaussianity via the Epps-Pulley test $T_("EP")$:

$
  T_("EP")(Z u) = N integral_(-inf)^(inf) |hat(phi)_N (t ; Z u) - phi_0 (t)|^2 w(t) d t
$

where $phi_0 (t) = e^(-0.5 t^2)$ is the theoretical Gaussian characteristic function, $hat(phi)_N (t ; Z u) = 1 / N sum_(j=1)^N e^(i t (Z u)_j)$ is the empirical characteristic function, and $w(t) = e^(-0.5 t^2)$ is the Gaussian weighting window.

The full SIGReg regularization loss is obtained by taking the average of $T_("EP")$ across all $M$ projection directions:

$ cal(L)_("SIGReg")(Z) = 1 / M sum_(m=1)^M T_("EP")(Z u_m) $

Compared to VICReg, this solution is simpler because there is only one hyperparameter and each encoder is independent so it works with multiple architectures. For complete PyTorch implementation details, see @fig-sigreg-code in Appendix A.


= Limitations and Practical Trade-Offs

While these methods effectively stabilize latent prediction training, each introduces distinct practical trade-offs identified in literature:

- **Expander Dimensionality Sensitivity in VICReg:** While Bardes et al. @bardes2022vicreg demonstrate that VICReg inherits a critical structural bottleneck from Barlow Twins @zbontar2021barlow: a heavy reliance on high-dimensional expander heads. Computing off-diagonal decorrelation across the covariance matrix $C in RR^(d times d)$ requires huge embedding dimensions ($d >= 8192$). Reducing the expander dimensionality to $d = 256$ causes top-1 accuracy to drop dramatically from $68.8\%$ to $55.9\%$ @bardes2022vicreg. Furthermore, balancing the three loss weighting parameters ($lambda, mu, nu$) requires careful hyperparameter tuning.

- **Topological Constraints of SIGReg:** Balestriero & LeCun @balestriero2025sigreg demonstrate that enforcing an isotropic Gaussian distribution $cal(N)(0, I)$ guarantees full effective rank ($"ER"(C) approx d$) and theoretically minimizes downstream prediction risk. However, as noted by recent analyses @weak_sigreg2026, forcing embeddings uniformly onto a spherical Gaussian manifold can over-constrain datasets that naturally inhabit highly non-Gaussian, multi-modal, or hierarchical topologies. Furthermore, computing $M$ random projections scales with embedding dimension $d$, requiring simplified sketching approximations for very high-dimensional layers @weak_sigreg2026.

- **Masking Heuristics in Spatial JEPAs:** Assran et al. @assran2023ijepa emphasize that multi-block masking strategies are not universally applicable, they are domain-specific. While large spatial blocks work well to prevent shortcuts in 2D images, applying similar logic to non-spatial or sequential modalities (e.g., continuous signals, graphs) is difficult. It requires delicate, task-specific redesigns to avoid trivial shortcut copies and the inability to reach target representations.

- **Instability in Momentum Encoders:** As shown by Chen & He @chen2021simsiam and Grill et al. @grill2020byol, relying on Exponential Moving Average (EMA) parameter updates ($tau approx 0.996$) creates an empirical heuristic. Not using an optimal parameter can lead to hidden representation degradation or slow convergence without explicit loss divergence warnings.

= Conclusion

JEPAs present a computationally efficient alternative to generative pretraining in self-supervised learning, but their latent-space objectives introduce inherent risks of representation collapse. Dimensional collapse, in particular, can silently degrade feature diversity and downstream generalization. Consequently, robust training of JEPAs necessitates deliberate mitigation strategies, ranging from architectural design choices, capacity bottlenecks, and masking strategies to explicit variance-covariance regularization and distributional constraints like SIGReg. Understanding the trade-offs of each method remains critical for scaling JEPAs to new modalities and complex real-world representations.

#bibliography("refs.bib")

= Appendix

== Appendix A: SIGReg Implementation Details

#block(
  width: 100% + 2 * 2.5cm, // Expands beyond the body text margins
  inset: (x: -2.5cm),
)[
  #figure(
    align(left)[
      #text(size: 7pt)[
        ```python
        def sigreg_loss(x, global_step, num_slices=256):
            """
            SIGReg PyTorch implementation adapted from Balestriero & LeCun (2025).
            Evaluates 1D non-Gaussianity via discretized trapezoidal integration.
            """
            dev = dict(device=x.device)
            g = torch.Generator(**dev).manual_seed(global_step)

            # Sample M random 1D unit projection directions
            A = torch.randn((x.size(1), num_slices), generator=g, **dev)
            A /= A.norm(p=2, dim=0)

            # Evaluate Empirical Characteristic Function over 17 grid points t
            t = torch.linspace(-5, 5, 17, **dev)
            exp_f = torch.exp(-0.5 * t**2)
            x_t = (x @ A).unsqueeze(2) * t                       # Shape: (N, M, T)
            ecf = (1j * x_t).exp().mean(0)                       # Shape: (M, T)

            # Weighted L2 distance via 17-point trapezoidal integration
            err = (ecf - exp_f).abs().square().mul(exp_f)
            return torch.trapz(err, t, dim=1).mean()
        ```]
    ],
    caption: [
      PyTorch implementation of the discretized Epps-Pulley test used in SIGReg, adapted directly from @balestriero2025sigreg.
    ],
  ) <fig-sigreg-code>
]
