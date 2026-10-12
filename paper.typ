#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/fletcher:0.5.5": diagram, edge, node
#import "@preview/cetz:0.3.2"

#show: ieee.with(
  title: [The Challenge of Representation Collapse in JEPA Models],
  abstract: [
    Joint Embedding Predictive Architectures (JEPAs) represent a significant advancement in self-supervised learning. Unlike traditional generative approaches that reconstruct inputs in the original modality (e.g., pixels), JEPAs predict representations directly in latent space. While JEPAs offer high sample efficiency and robustness compared to generative methods, they introduce a critical vulnerability: representation collapse. Preventing this failure mode is essential to ensure that learned embeddings remain informative and preserve fine-grained features for downstream tasks. This overview describes the JEPA framework, provides mathematical definitions of collapse, and outlines mitigation paradigms : including historical contrastive approaches, architectural asymmetry, masking strategies, explicit regularization, and state-of-the-art methods like SIGReg. Finally, we present their limitations discussed in the literature.
  ],
  authors: (
    (
      name: "Pierre-Emmanuel Baumier, Tanguy Paymal",
      department: [Pharmacy Faculty],
      organization: [UGA],
      location: [Grenoble, France],
      email: "pierre-emmanuel.baumier@etu.univ-grenoble-alpes.fr, tanguy.paymal@etu.univ-grenoble-alpes.fr",
    ),
  ),
  figure-supplement: [Fig.],
)

= Introduction



Deep learning in visual applications such as radiology and pathology relies heavily on strong feature representations. A representation is a numerical encoding of an input that captures features useful for subsequent tasks. Supervised approaches learn these representations from expert-annotated data. The effort required to obtain reliable labels motivates self-supervised learning (SSL), which learns from unlabeled observations @assran2023ijepa and avoids this bottleneck.

Generative self supervised models waste significant capacity reconstructing raw pixel details, such as high-frequency noise and irrelevant background variations, rather than learning high-level semantics. Joint Embedding Predictive Architectures (JEPAs) @lecun2022path overcome this by replacing raw signal reconstruction with direct prediction in a latent space @assran2023ijepa. By training the network to predict features of another view of the same underlying input, JEPAs can learn useful abstract representations. This motivates more computationally efficient prediction @assran2023ijepa and potentially smaller models: LeWorldModel, for example, uses 15 million parameters @lewm2026. Potential gains in model size and computation remain specific to the architecture and task. Which details they discard, and which distinctions they preserve, depend on the prediction task and training procedure.

However, predicting directly in latent space introduces a critical failure mode: *representation collapse*. When a model collapses, its encoder produces uninformative or degenerate embeddings, losing the discriminative capacity needed to differentiate features. This overview provides a concise structural analysis of the JEPA framework, formalizes the mechanics of representation collapse, and reviews leading mitigation paradigms alongside their practical trade-offs.



= Theoretical Architecture of JEPA (LeCun Framework)

Introduced by Yann LeCun @lecun2022path, the Joint-Embedding Predictive Architecture (JEPA), shown in @fig-lecun-jepa, processes paired views of an input signal through parallel encoding and prediction branches.

Architecturally, the model consists of three core components:
1. *Context Encoder ($E_theta_x$):* Maps an observable input region $x$ into a latent embedding $s_x$.
2. *Target Encoder ($E_theta_y$):* Maps a target or complementary view $y$ into an embedding $s_y$.
3. *Predictor ($P_phi$):* Takes $s_x$ alongside a conditioning latent variable $z$ (which encodes positional or transformation information) and produces an estimation $hat(s)_y$ of the target representation $s_y$.


#let c-in = rgb("#EBF3FA")
#let c-net = rgb("#FFFFFF")
#let c-emb = rgb("#F4ECF7")
#let c-z = rgb("#FEF5E7")
#let c-loss = rgb("#FDEDEC")
#let grey = rgb("#555555")

#figure(
  cetz.canvas({
    import cetz.draw: *
    set-style(
      rect: (radius: 0.06, stroke: 0.5pt),
      line: (stroke: 0.5pt),
      content: (padding: 1.5pt),
    )

    let arrow = (end: ">", size: 0.12)

    // Rectangular node (data / embeddings)
    let box(name, a, b, fill, body) = {
      rect(a, b, name: name, fill: fill)
      content(name, text(size: 7pt, body))
    }

    // Trapezoidal node (neural network)
    let trap(cx, y0, y1, wb, wt, body) = {
      line(
        (cx - wb / 2, y0),
        (cx + wb / 2, y0),
        (cx + wt / 2, y1),
        (cx - wt / 2, y1),
        close: true,
        fill: c-net,
      )
      content((cx, (y0 + y1) / 2), text(size: 7pt, body))
    }

    let xl = 0.0 // left column centre (context branch)
    let xr = 5.0 // right column centre (target branch)

    // Representation space (dashed region)
    rect(
      (-1.9, 2.4),
      (6.9, 6.0),
      stroke: (dash: "dashed", thickness: 0.5pt, paint: grey),
      fill: none,
      radius: 0.1,
    )
    content(
      (6.05, 4.9),
      text(size: 6pt, fill: grey, style: "italic")[Representation \ space],
    )

    // Inputs
    box("x", (xl - 1.6, 0.0), (xl + 1.6, 0.6), c-in)[Context $x$]
    box("y", (xr - 1.6, 0.0), (xr + 1.6, 0.6), c-in)[Target $y$]

    // Encoders
    trap(xl, 1.1, 2.1, 3.4, 2.4)[*Context Encoder*]
    trap(xr, 1.1, 2.1, 3.4, 2.4)[*Target Encoder*]

    // Embeddings
    box("sx", (xl - 1.2, 2.6), (xl + 1.2, 3.2), c-emb)[Embedding $s_x$]
    box("sy", (xr - 1.2, 2.6), (xr + 1.2, 3.2), c-emb)[Embedding $s_y$]

    // Predictor
    trap(xl, 3.7, 4.7, 3.2, 2.4)[*Predictor*]

    // Latent variable
    box("z", (1.8, 3.8), (4.0, 4.6), c-z)[*Latent Variable* \ $z$]

    // Predicted embedding
    box("shat", (xl - 1.2, 5.2), (xl + 1.2, 5.8), c-emb)[Prediction $hat(s)_y$]

    // Energy / loss
    box("loss", (xl - 1.6, 6.5), (xr + 1.6, 7.3), c-loss)[
      *Prediction Loss / Energy* \ $cal(E)(x, y) = D(hat(s)_y, s_y)$
    ]

    // Connections
    line((xl, 0.6), (xl, 1.1), mark: arrow)
    line((xr, 0.6), (xr, 1.1), mark: arrow)
    line((xl, 2.1), (xl, 2.6), mark: arrow)
    line((xr, 2.1), (xr, 2.6), mark: arrow)
    line((xl, 3.2), (xl, 3.7), mark: arrow)
    line("z.west", (1.4, 4.2), mark: arrow)
    line((xl, 4.7), (xl, 5.2), mark: arrow)
    line((xl, 5.8), (xl, 6.5), mark: arrow)
    line((xr, 3.2), (xr, 6.5), mark: arrow)
  }),
  caption: [
    Joint-Embedding Predictive Architecture (JEPA) @lecun2022path],
) <fig-lecun-jepa>

To train the model, the primary objective is to minimize a distance metric (or "energy") $cal(E)$ between the predicted target features $hat(s)_y$ and the actual target features $s_y$:

$
  cal(E)(x,y) = D(hat(s)_y, s_y) = norm(P_phi (E_(theta_x)(x), z) - E_(theta_y)(y))^2
$

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

Assuming zero-centered embeddings ($1/N sum_i Z_(i, :) = 0$).

Let $sigma_1 >= sigma_2 >= ... >= sigma_d >= 0$ be the singular values of $Z$.

== Complete Collapse

Complete collapse occurs when different inputs are mapped to the same representation. For example, a model may produce nearly identical embeddings for images of healthy tissue and images containing a lesion. The prediction loss can nevertheless be small if both the predicted and target representations have collapsed to the same constant.

$
  "Var"(Z_((:, j))) = 0, quad forall j in {1, ..., d} \
  => D(P_phi (E_(theta_x)(x), z), E_(theta_y)(y)) = 0
$

In this mode, the loss is minimized because the model has learned to ignore the input information, discarding all semantics.

== Dimensional Collapse

Dimensional collapse is subtler: embeddings retain some variation, but that variation is concentrated in only a small number of directions. The model may therefore distinguish certain features while losing others, even when its training loss remains stable. This loss of diversity can be investigated through the singular values of the embedding matrix $Z$ and summarized using its effective rank.

$
  "ER"(Z) = exp(- sum_(i=1)^d p_i log p_i), quad p_i = (sigma_i) / (sum_(j=1)^d sigma_j)
$

Intuitively, effective rank estimates how many dimensions contribute meaningfully to the representation. A low effective rank relative to the embedding dimension $d$ indicates that the embeddings use only a restricted portion of the available representation space.


= Methods to Prevent Representation Collapse

Preventing representation collapse in joint-embedding architectures (JEA) relies on distinct algorithmic and data strategies. They seek to preserve variation while learning agreement between representations :

== Contrastive Methods and Dual-Term Objectives

Historically, contrastive learning methods (e.g., SimCLR @chen2020simclr) prevented collapse in joint embeddings by using objectives like InfoNCE @oord2018infonce. These objectives pull positive pair embeddings together while explicitly repelling negative pairs. Non-contrastive approaches, including JEPA, instead rely on architectural asymmetries, prediction objectives, and training dynamics to learn informative representations without requiring explicit negative pairs.

== Architectural Asymmetry
Non-contrastive models introduce structural or dynamic asymmetries between processing branches to avoid identity shortcuts:

1. *Input Views and Masking Strategies*: Predicting representations of unobserved regions encourages the model to capture contextual information rather than copy visible details.
2. *Predictor asymmetry and Stop Gradient*: An additional prediction network and Stop gradient operation changes the optimization dynamics and can help prevent trivial solutions under particular conditions. Blocking backpropagation along the target encoder pathway breaks the joint optimization loop:
  $ cal(L) = D(P_phi (E_(theta_x)(x), z), "sg"[E_(theta_y)(y)]) $
  Gradients propagate only through $theta_x$ and $phi$, setting $nabla_(theta_y) cal(L) := 0$.
4. *Exponential Moving Average (EMA)*: Used in BYOL @grill2020byol and later used in I-JEPA @assran2023ijepa, the target encoder parameters $theta_y$ update gradually via a momentum combination with the weights from the encoder $theta_x$ to stabilize the training:
  $ theta_y^((t)) <- tau theta_y^((t-1)) + (1 - tau) theta_x^((t)) $

== Information Maximization and Variance-Covariance Regularization (VICReg)

VICReg @bardes2022vicreg avoids collapse by optimizing a loss with three distinct terms: an *Invariance criterion* $s(Z_x, Z_y)$, and explicit *Variance* $v(Z)$ and *Covariance* $c(Z)$ regularization penalties applied to both embedding sets $Z_x, Z_y in RR^(N times d)$:

$
  cal(L) & = lambda s(Z_x, Z_y) + mu [v(Z_x) + v(Z_y)] \
         & quad + nu [c(Z_x) + c(Z_y)]
$

where $lambda, mu, nu > 0$ are hyperparameter weighting coefficients.

1. *Invariance $s(Z_x, Z_y)$*: Encourages corresponding representations to remain similar.
2. *Variance $v(Z)$*: Prevents individual feature dimensions from becoming constant across the batch.
3. *Covariance $c(Z)$*: Penalizes correlations between different feature dimensions, reducing redundancy.

These moment constraints do not determine the complete embedding distribution, motivating the next approach.

== Sketched Isotropic Gaussian Regularization (SIGReg)

To prevent representation collapse, Balestriero & LeCun introduced SIGReg @balestriero2025sigreg, it regularizes the distribution of the learned embeddings toward an isotropic Gaussian. It approximates this constraint by projecting embeddings onto multiple random one-dimensional directions and penalizing deviations from the expected Gaussian distribution. Unlike VICReg, which explicitly controls feature variances and pairwise covariances, SIGReg evaluates a broader property of the representation distribution. The loss is then computed like so :
$
  cal(L)_"LeJEPA" = cal(L)_("pred") + lambda cal(L)_("SIGReg")
$
SIGReg loss is simpler because there is only one hyperparameter $lambda$ and each encoder is independent so it works with multiple architectures.


= Limitations and Practical Trade-Offs

Although these methods can stabilize representation learning, they introduce
different assumptions and practical trade-offs:

- *VICReg: Dimensionality and Hyperparameter Sensitivity.* VICReg relies on variance and covariance regularization to prevent collapse. Computing the covariance matrix $C in RR^(d times d)$ can become expensive for high-dimensional embeddings. Bardes et al. @bardes2022vicreg report that reducing the expander dimensionality from $d = 8192$ to $d = 256$ decreases top-1 accuracy from $68.8%$ to $55.9%$ in their experimental setting. Moreover, balancing the three loss weights ($lambda$, $mu$, $nu$) requires careful hyperparameter tuning, which can complicate practical deployment.

- *SIGReg: Distributional Assumptions.* SIGReg regularizes representations toward an isotropic Gaussian distribution, encouraging variation across feature dimensions. However, this assumption may not be equally suitable for every dataset or downstream task, particularly when the desired representations have complex, multimodal structure. Although SIGReg evaluates the distribution more broadly than pairwise covariance regularization, its effectiveness still depends on the task and training objective. Its use of $M$ random projections also introduces additional computation, especially for high-dimensional embeddings @balestriero2025sigreg.

- *Masking: Dependence on the Input Modality.* Masking strategies used in image-based JEPAs, such as I-JEPA @assran2023ijepa, rely on the spatial structure of images. Predicting large, masked regions can encourage semantic learning rather than local interpolation. However, the same strategy cannot necessarily be transferred directly to sequential data, graphs, or other non-spatial modalities. The masking strategy must be adapted to the structure of the data to avoid trivial prediction shortcuts or targets that are too difficult to predict.

- *Momentum Encoders: Training Sensitivity.* Exponential moving averages (EMA), used in methods such as BYOL @grill2020byol and I-JEPA @assran2023ijepa, update the target encoder gradually using the context encoder's parameters. This can stabilize training, but the momentum coefficient influences how quickly the target encoder adapts. An unsuitable value may slow convergence or affect representation quality, and these effects are not necessarily reflected by the prediction loss alone. EMA should therefore be understood as a practical stabilization mechanism rather than a universal guarantee against collapse.

= Conclusion

JEPAs present a computationally efficient alternative to generative pretraining in self-supervised learning, but their latent-space objectives introduce inherent risks of representation collapse. Dimensional collapse, in particular, can silently reduce feature diversity even if the prediction error is low. Consequently, robust training of JEPAs requires deliberate mitigation strategies, ranging from architectural choices, training dynamics, and explicit regularization methods such as VICReg and SIGReg. Successful representation learning needs balancing predictive consistency with sufficient diversity to preserve task-relevant information. Understanding the trade-offs of each method remains critical for scaling JEPAs to new modalities and complex real-world representations.

#bibliography("refs.bib")

