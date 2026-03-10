# Machine Learning Algorithms — Complete Reference Handbook

> A concise, structured quick-reference guide covering all major ML algorithms across Supervised and Unsupervised Learning.

---

# Table of Contents

1. [Supervised Learning](#1-supervised-learning)
   - [1.1 Regression Algorithms](#11-regression-algorithms)
   - [1.2 Classification Algorithms](#12-classification-algorithms)
2. [Unsupervised Learning](#2-unsupervised-learning)
   - [2.1 Clustering Algorithms](#21-clustering-algorithms)
   - [2.2 Dimensionality Reduction](#22-dimensionality-reduction)
3. [Comparison Tables](#3-comparison-tables)
4. [Practical Model Selection Guide](#4-practical-model-selection-guide)
5. [ML Pipeline Overview](#5-ml-pipeline-overview)

---

# 1. Supervised Learning

> Learns a mapping from inputs **X** → outputs **y** using labeled training data.

---

## 1.1 Regression Algorithms

> Predict a **continuous** numeric output.

---

### Linear Regression

- **Description:** Fits a straight line (or hyperplane) through data to minimize residual error.
- **Mathematical Intuition:** Minimizes the sum of squared differences between predicted and actual values: `ŷ = wᵀx + b`, optimized via Ordinary Least Squares (OLS) or Gradient Descent.
- **When to Use:** Linear relationships between features and target; baseline model.
- **Advantages:** Fast, interpretable, no hyperparameters (OLS), well-understood statistically.
- **Disadvantages:** Assumes linearity; sensitive to outliers; poor with multicollinearity.
- **Example Use Case:** Predicting house prices from square footage.

---

### Polynomial Regression

- **Description:** Extends linear regression by adding polynomial feature terms (x², x³, …).
- **Mathematical Intuition:** `ŷ = w₀ + w₁x + w₂x² + … + wₙxⁿ` — still linear in coefficients, nonlinear in input space.
- **When to Use:** Non-linear relationships that are smooth and continuous.
- **Advantages:** Captures curvature; still interpretable.
- **Disadvantages:** Prone to overfitting at high degrees; sensitive to outliers.
- **Example Use Case:** Modeling population growth curves.

---

### Ridge Regression (L2)

- **Description:** Linear regression with L2 regularization to penalize large weights.
- **Mathematical Intuition:** Minimizes `||y - Xw||² + α||w||²`. Shrinks coefficients toward zero but never exactly to zero.
- **When to Use:** When multicollinearity is present; many features with small effects.
- **Advantages:** Reduces overfitting; stable with correlated features.
- **Disadvantages:** Does not perform feature selection; all features retained.
- **Example Use Case:** Predicting stock returns with many correlated financial indicators.

---

### Lasso Regression (L1)

- **Description:** Linear regression with L1 regularization that induces sparsity.
- **Mathematical Intuition:** Minimizes `||y - Xw||² + α||w||₁`. Drives irrelevant feature weights exactly to zero.
- **When to Use:** High-dimensional data; automatic feature selection is desired.
- **Advantages:** Sparse model; built-in feature selection; interpretable.
- **Disadvantages:** Struggles when features are highly correlated; selects only one arbitrarily.
- **Example Use Case:** Genomics — selecting relevant genes from thousands.

---

### Elastic Net

- **Description:** Combines L1 (Lasso) and L2 (Ridge) penalties.
- **Mathematical Intuition:** `Loss + α·ρ||w||₁ + α·(1-ρ)/2·||w||²` where ρ balances L1/L2.
- **When to Use:** Many correlated features; want both sparsity and stability.
- **Advantages:** Best of Ridge + Lasso; handles correlated groups of features.
- **Disadvantages:** Two hyperparameters (α, ρ) to tune.
- **Example Use Case:** Text regression with highly correlated word features.

---

### Decision Tree Regression

- **Description:** Recursively partitions feature space into regions, predicting the mean of each region.
- **Mathematical Intuition:** Splits chosen to minimize variance (MSE) within child nodes at each step.
- **When to Use:** Non-linear data; need interpretability; quick prototyping.
- **Advantages:** No scaling needed; handles mixed feature types; interpretable.
- **Disadvantages:** High variance; prone to overfitting; unstable.
- **Example Use Case:** Predicting insurance claims based on age, region, and history.

---

### Random Forest Regression

- **Description:** Ensemble of many decision trees trained on random subsets of data and features (bagging).
- **Mathematical Intuition:** Final prediction = average of all tree predictions. Variance reduced by averaging uncorrelated trees.
- **When to Use:** Non-linear relationships; tabular data; need robustness.
- **Advantages:** Robust to overfitting; handles missing data well; feature importance scores.
- **Disadvantages:** Less interpretable than single tree; slower training; large memory footprint.
- **Example Use Case:** Predicting energy consumption across buildings.

---

### Gradient Boosting Regression

- **Description:** Sequentially trains trees where each corrects the residual errors of its predecessor.
- **Mathematical Intuition:** `F_m(x) = F_{m-1}(x) + η · h_m(x)` where `h_m` fits the negative gradient of the loss.
- **When to Use:** High-accuracy tabular regression; can handle mixed feature types.
- **Advantages:** State-of-the-art accuracy on tabular data; flexible loss functions.
- **Disadvantages:** Slow training; many hyperparameters; can overfit small datasets.
- **Example Use Case:** Predicting sales revenue with complex interactions.

---

### XGBoost

- **Description:** Optimized, regularized gradient boosting with second-order gradient approximation and tree pruning.
- **Mathematical Intuition:** Adds L1/L2 regularization to the boosting objective; uses Taylor expansion for efficient optimization.
- **When to Use:** Competitions; large tabular datasets; need speed and accuracy.
- **Advantages:** Fast; handles sparse data natively; built-in regularization; parallel computation.
- **Disadvantages:** Many hyperparameters; less interpretable; memory-intensive for very large datasets.
- **Example Use Case:** Click-through rate prediction in advertising.

---

### LightGBM

- **Description:** Gradient boosting using histogram-based splitting and leaf-wise tree growth.
- **Mathematical Intuition:** Grows tree leaf-wise (choosing the leaf with max loss reduction) rather than level-wise, enabling faster convergence.
- **When to Use:** Very large datasets; need fast training; high-cardinality categoricals.
- **Advantages:** Fastest among boosting libraries; low memory; handles categorical features natively.
- **Disadvantages:** Can overfit small datasets; leaf-wise growth can be unstable.
- **Example Use Case:** Real-time fraud scoring on millions of transactions.

---

### CatBoost

- **Description:** Gradient boosting optimized for categorical features using ordered boosting to prevent target leakage.
- **Mathematical Intuition:** Uses ordered statistics to encode categoricals without leakage; symmetric decision trees for efficiency.
- **When to Use:** Datasets with many categorical features; minimal preprocessing desired.
- **Advantages:** Handles categoricals automatically; robust to overfitting; minimal tuning needed.
- **Disadvantages:** Slower training than LightGBM; higher memory usage.
- **Example Use Case:** Predicting customer lifetime value from CRM data.

---

### Support Vector Regression (SVR)

- **Description:** Finds a function within an ε-tube of the training data while maximizing the margin.
- **Mathematical Intuition:** Minimizes `½||w||²` subject to `|y_i - f(x_i)| ≤ ε`. Uses kernel trick for nonlinear regression.
- **When to Use:** Small-to-medium datasets; nonlinear relationships; robust to outliers (with ε-insensitive loss).
- **Advantages:** Effective in high dimensions; kernel flexibility; outlier-robust.
- **Disadvantages:** Does not scale well (O(n²–n³)); sensitive to feature scaling; hard to interpret.
- **Example Use Case:** Predicting protein concentrations from spectroscopic data.

---

### KNN Regression

- **Description:** Predicts the output as the average of the k nearest neighbors in feature space.
- **Mathematical Intuition:** `ŷ = (1/k) Σ y_i` for the k closest training points (by Euclidean or other distance).
- **When to Use:** Small datasets; no strong parametric assumptions; local structure matters.
- **Advantages:** Simple; no training phase; naturally handles multi-output regression.
- **Disadvantages:** Slow at inference (O(n)); sensitive to irrelevant features and scale; suffers in high dimensions.
- **Example Use Case:** Recommending product prices based on similar past transactions.

---

### Bayesian Regression

- **Description:** Places a prior distribution over model parameters and updates it using Bayes' theorem to obtain a posterior.
- **Mathematical Intuition:** `P(w|X,y) ∝ P(y|X,w) · P(w)`. Predictions are distributions, not point estimates.
- **When to Use:** Small data; need uncertainty quantification; incorporating prior knowledge.
- **Advantages:** Full uncertainty estimates; principled regularization via priors; robust to overfitting.
- **Disadvantages:** Computationally expensive; requires prior specification; harder to implement.
- **Example Use Case:** Clinical trial outcome prediction with small patient cohorts.

---

## 1.2 Classification Algorithms

> Predict a **discrete class label** from input features.

---

### Logistic Regression

- **Description:** Linear model for binary (or multiclass) classification using a sigmoid activation.
- **Core Idea:** `P(y=1|x) = σ(wᵀx + b)` — decision boundary is a hyperplane; trained by maximizing log-likelihood.
- **When to Use:** Binary/multiclass classification; need probability outputs; interpretable baseline.
- **Advantages:** Fast; interpretable coefficients; calibrated probabilities; works well linearly separable data.
- **Disadvantages:** Assumes linear decision boundary; poor on complex nonlinear data.
- **Example Use Case:** Email spam detection.

---

### K-Nearest Neighbors (KNN)

- **Description:** Classifies a point based on majority vote of its k nearest neighbors.
- **Core Idea:** Non-parametric; stores all training data; no explicit training — just distance computation at inference.
- **When to Use:** Small datasets; non-linear boundaries; interpretable local decisions.
- **Advantages:** Simple; no assumptions about data distribution; naturally multi-class.
- **Disadvantages:** Slow inference; sensitive to irrelevant features and scale; struggles in high dimensions.
- **Example Use Case:** Handwritten digit recognition on small datasets.

---

### Support Vector Machine (SVM)

- **Description:** Finds the hyperplane that maximally separates classes; uses kernel trick for non-linear boundaries.
- **Core Idea:** Maximize margin between support vectors. With kernel K(x,x'), maps to higher-dimensional space implicitly.
- **When to Use:** High-dimensional data; text classification; small-to-medium datasets; clear margin of separation.
- **Advantages:** Effective in high dimensions; memory-efficient (only support vectors stored); versatile kernels.
- **Disadvantages:** Slow on large datasets; sensitive to feature scale; no direct probability output (requires Platt scaling).
- **Example Use Case:** Image classification, bioinformatics (gene expression).

---

### Decision Tree

- **Description:** Recursively splits data using feature thresholds to create a tree of decision rules.
- **Core Idea:** At each node, chooses the split that maximizes information gain (entropy) or minimizes Gini impurity.
- **When to Use:** Need interpretability; mixed feature types; non-linear boundaries; quick baseline.
- **Advantages:** Interpretable (white-box); no feature scaling needed; handles categorical data.
- **Disadvantages:** Prone to overfitting; unstable (small data changes → different trees).
- **Example Use Case:** Loan approval decision systems.

---

### Random Forest

- **Description:** Ensemble of decorrelated decision trees via bagging + random feature subsets.
- **Core Idea:** Each tree votes; majority class wins. Diversity via random subsampling reduces variance.
- **When to Use:** General-purpose tabular classification; need feature importance; robust performance.
- **Advantages:** Resistant to overfitting; feature importance; handles missing values; parallelizable.
- **Disadvantages:** Black-box; slow for real-time inference; large memory footprint.
- **Example Use Case:** Medical diagnosis from patient records.

---

### Gradient Boosting (GBM)

- **Description:** Sequentially builds trees, each correcting the errors of the previous ensemble.
- **Core Idea:** Minimizes a differentiable loss by fitting new trees to the negative gradient (pseudo-residuals).
- **When to Use:** Tabular data competitions; high accuracy requirements; complex feature interactions.
- **Advantages:** State-of-the-art on tabular data; flexible loss functions; handles mixed types.
- **Disadvantages:** Slow training; risk of overfitting; many hyperparameters.
- **Example Use Case:** Customer churn prediction.

---

### XGBoost

- **Description:** Regularized, optimized gradient boosting with parallel tree construction and pruning.
- **Core Idea:** Uses second-order Taylor expansion of loss + L1/L2 regularization. Efficient sparse-aware split-finding.
- **When to Use:** Large datasets; high-accuracy classification; structured/tabular data.
- **Advantages:** Speed, accuracy, regularization, handles missing values natively.
- **Disadvantages:** Many hyperparameters; memory-heavy; less suited for unstructured data.
- **Example Use Case:** Kaggle competitions; credit scoring.

---

### LightGBM

- **Description:** Leaf-wise gradient boosting with histogram-based feature binning.
- **Core Idea:** Grows the leaf with the greatest loss reduction; GOSS sampling and EFB bundling for speed.
- **When to Use:** Very large datasets; speed-critical applications; high-cardinality categoricals.
- **Advantages:** Fastest training among boosting algorithms; low memory; native categorical support.
- **Disadvantages:** Can overfit on small data; less stable than level-wise trees.
- **Example Use Case:** Real-time bidding and ad ranking.

---

### CatBoost

- **Description:** Gradient boosting with ordered encoding for categorical features to avoid target leakage.
- **Core Idea:** Uses oblivious symmetric trees and ordered boosting statistics for categorical variables.
- **When to Use:** Datasets heavy in categorical variables; want minimal preprocessing.
- **Advantages:** Best-in-class for categorical data; minimal tuning; robust to overfitting.
- **Disadvantages:** Slower than LightGBM; higher RAM; slower inference vs. XGBoost.
- **Example Use Case:** E-commerce recommendation and ranking.

---

### Naive Bayes

> A family of probabilistic classifiers based on Bayes' theorem with feature independence assumptions.

#### Gaussian Naive Bayes
- **Description:** Assumes continuous features follow a Gaussian (normal) distribution within each class.
- **Core Idea:** `P(x_i|y) = N(μ_y, σ²_y)`. Estimates mean and variance per class per feature.
- **When to Use:** Continuous features; real-valued sensor data.
- **Advantages:** Fast; works well with small data; handles real-valued features.
- **Disadvantages:** Strong independence assumption; poor if features are correlated.
- **Example Use Case:** Medical diagnosis with continuous measurements.

#### Multinomial Naive Bayes
- **Description:** Models feature counts (e.g., word frequencies); suited for discrete count data.
- **Core Idea:** `P(x_i|y) = (count(x_i, y) + α) / (total_count(y) + α·|V|)` — uses Laplace smoothing.
- **When to Use:** Text classification with bag-of-words or TF features.
- **Advantages:** Extremely fast; works well for text; interpretable.
- **Disadvantages:** Assumes features are counts (non-negative integers).
- **Example Use Case:** Spam filtering, topic classification.

#### Bernoulli Naive Bayes
- **Description:** Designed for binary/boolean features (feature present or absent).
- **Core Idea:** `P(x_i|y) = P_i^{x_i} · (1-P_i)^{1-x_i}` — penalizes absence of features unlike Multinomial.
- **When to Use:** Binary feature vectors (e.g., word occurrence, not frequency).
- **Advantages:** Penalizes absence of features; good for short texts.
- **Disadvantages:** Discards frequency information.
- **Example Use Case:** Sentiment classification with binary word presence features.

---

### Neural Networks (MLP)

- **Description:** Multi-layer feedforward network with nonlinear activation functions trained via backpropagation.
- **Core Idea:** Stacks linear transformations + nonlinearities: `h = σ(Wx + b)`. Universal function approximator.
- **When to Use:** Complex, high-dimensional data (images, text, audio); large datasets; non-tabular data.
- **Advantages:** Learns complex representations; state-of-the-art for unstructured data; scalable.
- **Disadvantages:** Requires large data; computationally expensive; black-box; sensitive to hyperparameters.
- **Example Use Case:** Image recognition, NLP, speech recognition.

---

### AdaBoost

- **Description:** Boosting algorithm that combines weak learners (stumps) by re-weighting misclassified samples.
- **Core Idea:** Each subsequent classifier focuses more on previously misclassified points; final prediction = weighted vote.
- **When to Use:** Binary classification; clean data (sensitive to noise/outliers); need interpretable ensemble.
- **Advantages:** Simple; reduces bias; less prone to overfitting than a single tree.
- **Disadvantages:** Sensitive to noisy data and outliers; slower than Random Forest.
- **Example Use Case:** Face detection (Viola-Jones framework).

---

# 2. Unsupervised Learning

> Finds patterns and structure in **unlabeled** data.

---

## 2.1 Clustering Algorithms

> Groups similar data points together without predefined labels.

---

### K-Means

- **Description:** Partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids.
- **Core Idea:** Minimize within-cluster sum of squared distances (inertia). Uses Expectation-Maximization style updates.
- **When to Use:** Large datasets; roughly spherical, equal-sized clusters; known K.
- **Advantages:** Fast (O(nKd)); scalable; simple; widely available.
- **Disadvantages:** Requires K upfront; assumes spherical clusters; sensitive to outliers and initialization.

---

### K-Medoids (PAM)

- **Description:** Like K-Means but uses actual data points (medoids) as cluster centers.
- **Core Idea:** Minimize sum of dissimilarities to medoid. Medoid = data point minimizing within-cluster distance sum.
- **When to Use:** Non-Euclidean distances; need robust clustering; medoids should be real data points.
- **Advantages:** Robust to outliers; works with any distance metric.
- **Disadvantages:** Computationally expensive O(k(n-k)²); slower than K-Means.

---

### Hierarchical Clustering

- **Description:** Builds a tree (dendrogram) of clusters by iteratively merging (agglomerative) or splitting (divisive) groups.
- **Core Idea:** Agglomerative: start with n clusters, merge closest pair at each step. Linkage criteria: single, complete, average, Ward.
- **When to Use:** Unknown K; want to explore cluster hierarchy; small-to-medium datasets.
- **Advantages:** No K required; produces dendrogram; works with any distance.
- **Disadvantages:** O(n² log n) time; cannot undo merges; sensitive to noise.

---

### DBSCAN

- **Description:** Density-Based Spatial Clustering of Applications with Noise. Groups dense regions, labels sparse points as noise.
- **Core Idea:** A point is a core point if ≥ MinPts neighbors are within ε. Clusters = connected regions of core points.
- **When to Use:** Arbitrary-shaped clusters; noise/outlier detection; unknown K.
- **Advantages:** Finds clusters of arbitrary shape; robust to outliers; no K needed.
- **Disadvantages:** Sensitive to ε and MinPts; struggles with varying-density clusters; not scalable to very high dimensions.

---

### HDBSCAN

- **Description:** Hierarchical extension of DBSCAN using a stability-based cluster extraction from the cluster hierarchy.
- **Core Idea:** Builds a hierarchy of clusters across all density levels; extracts persistent clusters by maximizing stability.
- **When to Use:** Varying-density clusters; robust noise detection; when DBSCAN requires too much parameter tuning.
- **Advantages:** Handles varying densities; single key parameter (min_cluster_size); soft clustering available.
- **Disadvantages:** Higher computational cost than DBSCAN; more complex implementation.

---

### Mean Shift

- **Description:** Non-parametric algorithm that shifts each data point toward the region of highest local density.
- **Core Idea:** Iteratively moves points toward the mean of points within a kernel bandwidth window until convergence.
- **When to Use:** Unknown K; blob-shaped clusters; image segmentation.
- **Advantages:** Automatically finds K; robust to outliers; no cluster shape assumption.
- **Disadvantages:** Slow O(n²); bandwidth selection is critical; not scalable to large datasets.

---

### Gaussian Mixture Models (GMM)

- **Description:** Probabilistic model assuming data is generated from a mixture of K Gaussian distributions.
- **Core Idea:** Fits K Gaussians via Expectation-Maximization (E-step: compute soft assignments; M-step: update parameters).
- **When to Use:** Soft/probabilistic cluster assignments; elliptical clusters; density estimation.
- **Advantages:** Soft assignments; models cluster covariance; principled probabilistic framework.
- **Disadvantages:** Assumes Gaussian clusters; sensitive to initialization; requires K; can converge to local optima.

---

### Spectral Clustering

- **Description:** Uses eigenvalues of a graph Laplacian constructed from a similarity matrix to perform dimensionality reduction before clustering.
- **Core Idea:** Build affinity graph → compute Laplacian → take top-k eigenvectors → cluster with K-Means in eigenspace.
- **When to Use:** Non-convex clusters; graph/network data; manifold structure.
- **Advantages:** Can find non-convex clusters; uses global structure of data.
- **Disadvantages:** Expensive O(n³) eigendecomposition; requires K; large memory for affinity matrix.

---

### BIRCH

- **Description:** Balanced Iterative Reducing and Clustering using Hierarchies — builds a CF-Tree (Clustering Feature Tree) for summarizing data.
- **Core Idea:** Incrementally builds a compact summary of data (CF-Tree); final clustering done on leaf nodes.
- **When to Use:** Very large datasets; streaming data; memory-constrained environments.
- **Advantages:** Single pass (O(n)); handles large datasets; incremental/online learning.
- **Disadvantages:** Assumes spherical clusters; sensitive to threshold parameter; not great for high-dimensional data.

---

### Affinity Propagation

- **Description:** Passes "responsibility" and "availability" messages between data points to identify exemplars.
- **Core Idea:** Every point is a potential exemplar; messages converge to identify which points best represent clusters.
- **When to Use:** Unknown K; want algorithm to determine number of clusters; small-to-medium datasets.
- **Advantages:** Automatically finds K; exemplar-based (real data points as centers).
- **Disadvantages:** O(n²) memory and time; slow; can produce too many clusters; sensitive to preference parameter.

---

## 2.2 Dimensionality Reduction

> Reduces the number of features while preserving important structure.

---

### PCA (Principal Component Analysis)

- **Description:** Projects data onto orthogonal axes of maximum variance, ordered by explained variance.
- **Core Idea:** Computes eigenvectors of the covariance matrix; projects data onto top-k eigenvectors (principal components).
- **When to Use:** Linear dimensionality reduction; visualization; noise reduction; preprocessing before ML.
- **Advantages:** Fast; deterministic; interpretable components; removes correlated features.
- **Disadvantages:** Linear only; components may not be interpretable semantically; loses non-linear structure.

---

### Kernel PCA

- **Description:** Extends PCA to nonlinear manifolds using the kernel trick.
- **Core Idea:** Implicitly maps data to a high-dimensional feature space via kernel K(x,x'), then applies PCA there.
- **When to Use:** Non-linear structure in data; manifold data; when standard PCA fails.
- **Advantages:** Captures non-linear variance; flexible kernels (RBF, poly, etc.).
- **Disadvantages:** O(n²) memory; no explicit inverse transform; kernel choice matters.

---

### LDA (Linear Discriminant Analysis)

- **Description:** Supervised dimensionality reduction that maximizes between-class separability while minimizing within-class scatter.
- **Core Idea:** Finds projection axes that maximize `S_B / S_W` (between-class to within-class scatter ratio).
- **When to Use:** Preprocessing for classification; want maximally class-discriminative features.
- **Advantages:** Supervised (uses labels); maximally class-separable; at most C-1 components (C = classes).
- **Disadvantages:** Assumes Gaussian class distributions; linear only; requires labeled data.

---

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

- **Description:** Non-linear dimensionality reduction optimized for visualizing high-dimensional data in 2D/3D.
- **Core Idea:** Preserves local neighborhoods by modeling pairwise similarities as probabilities; minimizes KL divergence between high-D and low-D distributions.
- **When to Use:** Visualization of high-dimensional data; exploring cluster structure.
- **Advantages:** Excellent at revealing local cluster structure; widely used for visualization.
- **Disadvantages:** No out-of-sample extension; slow O(n²); non-deterministic; global structure not preserved.

---

### UMAP (Uniform Manifold Approximation and Projection)

- **Description:** Fast, scalable non-linear dimensionality reduction based on Riemannian geometry and fuzzy topology.
- **Core Idea:** Constructs a fuzzy topological representation of the high-D manifold and optimizes a low-D embedding to match it.
- **When to Use:** Visualization; preprocessing; faster alternative to t-SNE; preserves more global structure.
- **Advantages:** Faster than t-SNE; supports out-of-sample projection; preserves global structure better.
- **Disadvantages:** Hyperparameter sensitive (n_neighbors, min_dist); harder to interpret theoretically.

---

### ICA (Independent Component Analysis)

- **Description:** Decomposes signals into statistically independent non-Gaussian components (unlike PCA which finds uncorrelated components).
- **Core Idea:** `X = AS` — finds mixing matrix A such that components S are maximally non-Gaussian and independent.
- **When to Use:** Signal separation (blind source separation); feature extraction from mixed signals.
- **Advantages:** Finds statistically independent components; great for signal unmixing.
- **Disadvantages:** Cannot determine ordering or scaling of components; requires non-Gaussian data.

---

### Autoencoders

- **Description:** Neural network trained to compress input into a low-dimensional bottleneck (encoder) and reconstruct it (decoder).
- **Core Idea:** Minimize reconstruction loss `||x - decoder(encoder(x))||²`. Bottleneck layer = compressed representation.
- **When to Use:** Non-linear DR; anomaly detection; generative modeling; image compression.
- **Advantages:** Captures highly non-linear structure; flexible architecture; can be extended (VAE, denoising).
- **Disadvantages:** Requires large data; computationally expensive; black-box; hyperparameter-heavy.

---

### Factor Analysis

- **Description:** Statistical model assuming observed variables are linear combinations of latent (unobserved) factors plus noise.
- **Core Idea:** `X = ΛF + ε` where Λ = factor loadings, F = latent factors, ε = per-feature noise. Fits via EM.
- **When to Use:** Psychometrics; survey data; when noise is heteroskedastic per variable.
- **Advantages:** Models per-variable noise; interpretable factors; principled statistical framework.
- **Disadvantages:** Assumes linearity; factor rotation needed for interpretability; sensitive to model specification.

---

### SVD (Singular Value Decomposition)

- **Description:** Factorizes a matrix into `X = UΣVᵀ`. Truncated SVD keeps top-k singular values for compression.
- **Core Idea:** Equivalent to PCA on centered data but also applicable to sparse/non-square matrices (e.g., TF-IDF matrices).
- **When to Use:** Text data (LSA); collaborative filtering; matrix approximation; PCA on sparse matrices.
- **Advantages:** Works on sparse matrices; powerful for text; foundation of many algorithms.
- **Disadvantages:** Linear; no out-of-sample extension without projection; memory-intensive for dense matrices.

---

# 3. Comparison Tables

## Regression Algorithms

| Algorithm | Type | Handles Nonlinearity | Interpretable | Scalable | Typical Use Case |
|---|---|---|---|---|---|
| Linear Regression | Parametric | ❌ | ✅ High | ✅ | Baseline numeric prediction |
| Polynomial Regression | Parametric | ✅ Limited | ✅ Medium | ⚠️ | Smooth curve fitting |
| Ridge Regression | Parametric | ❌ | ✅ High | ✅ | Multicollinear features |
| Lasso Regression | Parametric | ❌ | ✅ High | ✅ | Feature selection |
| Elastic Net | Parametric | ❌ | ✅ Medium | ✅ | Correlated sparse features |
| Decision Tree | Non-parametric | ✅ | ✅ High | ⚠️ | Rule-based prediction |
| Random Forest | Ensemble | ✅ | ⚠️ Medium | ✅ | General tabular regression |
| Gradient Boosting | Ensemble | ✅ | ⚠️ Low | ⚠️ | High-accuracy tabular data |
| XGBoost | Ensemble | ✅ | ⚠️ Low | ✅ | Structured data, competitions |
| LightGBM | Ensemble | ✅ | ⚠️ Low | ✅✅ | Very large datasets |
| CatBoost | Ensemble | ✅ | ⚠️ Low | ✅ | Categorical-heavy data |
| SVR | Kernel | ✅ | ❌ | ⚠️ | Small, high-dimensional data |
| KNN Regression | Instance-based | ✅ | ✅ High | ❌ | Local pattern data |
| Bayesian Regression | Probabilistic | ❌ | ✅ High | ⚠️ | Uncertainty quantification |

---

## Classification Algorithms

| Algorithm | Type | Handles Nonlinearity | Interpretable | Scalable | Typical Use Case |
|---|---|---|---|---|---|
| Logistic Regression | Parametric | ❌ | ✅ High | ✅ | Binary classification baseline |
| KNN | Instance-based | ✅ | ✅ Medium | ❌ | Small-scale classification |
| SVM | Kernel | ✅ (kernel) | ⚠️ Low | ⚠️ | Text, high-dimensional data |
| Decision Tree | Non-parametric | ✅ | ✅ High | ⚠️ | Rule-based classification |
| Random Forest | Ensemble | ✅ | ⚠️ Medium | ✅ | General classification |
| Gradient Boosting | Ensemble | ✅ | ⚠️ Low | ⚠️ | High-accuracy tabular |
| XGBoost | Ensemble | ✅ | ⚠️ Low | ✅ | Competitions, credit scoring |
| LightGBM | Ensemble | ✅ | ⚠️ Low | ✅✅ | Large-scale classification |
| CatBoost | Ensemble | ✅ | ⚠️ Low | ✅ | Categorical-rich data |
| Gaussian NB | Probabilistic | ❌ | ✅ High | ✅ | Continuous feature classification |
| Multinomial NB | Probabilistic | ❌ | ✅ High | ✅ | Text classification |
| Bernoulli NB | Probabilistic | ❌ | ✅ High | ✅ | Binary feature classification |
| MLP (Neural Net) | Deep Learning | ✅✅ | ❌ | ✅✅ | Images, text, complex tasks |
| AdaBoost | Ensemble | ✅ | ⚠️ Medium | ⚠️ | Binary classification |

---

## Clustering Algorithms

| Algorithm | Type | Handles Nonlinearity | Requires K | Handles Noise | Scalable | Typical Use Case |
|---|---|---|---|---|---|---|
| K-Means | Partitional | ❌ | ✅ Yes | ❌ | ✅ | Customer segmentation |
| K-Medoids | Partitional | ❌ | ✅ Yes | ⚠️ | ❌ | Robust clustering |
| Hierarchical | Hierarchical | ⚠️ | ❌ No | ⚠️ | ❌ | Taxonomy, biology |
| DBSCAN | Density-based | ✅ | ❌ No | ✅ | ⚠️ | Geospatial clustering |
| HDBSCAN | Density-based | ✅ | ❌ No | ✅✅ | ⚠️ | Variable density data |
| Mean Shift | Density-based | ✅ | ❌ No | ⚠️ | ❌ | Image segmentation |
| GMM | Probabilistic | ⚠️ | ✅ Yes | ⚠️ | ⚠️ | Soft probabilistic clusters |
| Spectral | Graph-based | ✅ | ✅ Yes | ⚠️ | ❌ | Manifold/graph data |
| BIRCH | Hierarchical | ❌ | ❌ No | ⚠️ | ✅✅ | Large-scale streaming |
| Affinity Propagation | Message passing | ⚠️ | ❌ No | ⚠️ | ❌ | Unknown K, small data |

---

## Dimensionality Reduction Techniques

| Algorithm | Type | Handles Nonlinearity | Supervised | Scalable | Typical Use Case |
|---|---|---|---|---|---|
| PCA | Linear | ❌ | ❌ | ✅ | General preprocessing |
| Kernel PCA | Kernel | ✅ | ❌ | ❌ | Non-linear manifold data |
| LDA | Linear | ❌ | ✅ Yes | ✅ | Pre-classification reduction |
| t-SNE | Non-linear | ✅ | ❌ | ❌ | 2D/3D visualization |
| UMAP | Non-linear | ✅ | ❌ | ✅ | Visualization + preprocessing |
| ICA | Linear | ❌ | ❌ | ✅ | Signal source separation |
| Autoencoders | Deep Learning | ✅✅ | ❌ | ✅ | Complex non-linear DR |
| Factor Analysis | Probabilistic | ❌ | ❌ | ⚠️ | Latent factor discovery |
| SVD | Linear | ❌ | ❌ | ✅ | Text (LSA), recommendations |

---

# 4. Practical Model Selection Guide

## When to Choose Linear Models
- Relationship between features and target is approximately linear.
- Interpretability is critical (healthcare, finance, regulatory contexts).
- Dataset is small or medium-sized; training speed matters.
- Features are already well-engineered and informative.
- Use as a **fast baseline** before trying complex models.
- **Algorithms:** Linear/Logistic Regression, Ridge, Lasso, Elastic Net, LDA.

## When to Choose Tree-Based Models
- Mixed feature types (numerical + categorical) with minimal preprocessing.
- Non-linear relationships and interactions between features.
- Need interpretable rules and feature importance.
- Robust performance without extensive hyperparameter tuning.
- **Algorithms:** Decision Tree, Random Forest.

## When to Choose Ensemble Methods
- Maximum predictive accuracy on tabular/structured data is the goal.
- Large datasets with complex feature interactions.
- You can afford longer training times.
- Data is structured; no spatial/sequential patterns to exploit.
- **Algorithms:** Gradient Boosting, XGBoost, LightGBM, CatBoost, AdaBoost.
- **Rule of thumb:** XGBoost/LightGBM/CatBoost are the default choice for Kaggle-style tabular problems.

## When to Use Kernel Methods
- Small-to-medium dataset; high-dimensional feature space.
- Need to capture non-linear boundaries without deep learning.
- Text or bioinformatics data where custom kernels are meaningful.
- **Algorithms:** SVM, SVR, Kernel PCA.

## When to Use Neural Networks
- Data is unstructured: images, text, audio, video.
- Very large datasets (10k+ samples) available.
- Feature engineering is impractical — let the network learn representations.
- State-of-the-art performance is required.
- **Algorithms:** MLP, CNN, RNN, Transformer (task-dependent).
- **Caution:** Avoid NNs for small tabular datasets — ensemble trees usually win.

---

# 5. ML Pipeline Overview

## 1. Data Collection
Gather raw data from databases, APIs, sensors, or web scraping. Ensure coverage of all relevant conditions. Document data provenance and collection methodology.

## 2. Data Preprocessing
- **Handle missing values:** Imputation (mean/median/mode/KNN) or removal.
- **Remove duplicates** and fix inconsistencies.
- **Encode categoricals:** Label encoding, one-hot encoding, target encoding.
- **Scale features:** StandardScaler (z-score), MinMaxScaler, RobustScaler (for outliers).
- **Handle outliers:** Clip, transform (log), or use robust algorithms.

## 3. Feature Engineering
- **Create new features** from domain knowledge (ratios, interactions, aggregates).
- **Feature selection:** Filter methods (correlation, chi²), wrapper methods (RFE), embedded methods (Lasso, tree importance).
- **Dimensionality reduction:** PCA/UMAP for high-dimensional data.
- Goal: maximize signal, minimize noise and redundancy.

## 4. Model Training
- Split data: **Train / Validation / Test** (e.g., 70/15/15) or use **K-Fold Cross-Validation**.
- Train model on training set; monitor validation performance.
- Track experiments (MLflow, W&B).

## 5. Evaluation Metrics

| Task | Primary Metrics |
|---|---|
| Regression | MAE, RMSE, R², MAPE |
| Binary Classification | Accuracy, F1-Score, AUC-ROC, Precision, Recall |
| Multi-class Classification | Macro/Micro F1, Confusion Matrix, Top-k Accuracy |
| Clustering | Silhouette Score, Davies-Bouldin, Calinski-Harabasz, Adjusted Rand Index |
| Ranking | NDCG, MAP, MRR |

## 6. Hyperparameter Tuning
- **Grid Search:** Exhaustive; good for small search spaces.
- **Random Search:** Faster; samples randomly; often better than grid search.
- **Bayesian Optimization (Optuna, Hyperopt):** Learns from past trials; most efficient.
- **Early Stopping:** Prevents overfitting in iterative models (GBM, NNs).
- Always tune on **validation set**, never on test set.

## 7. Deployment
- **Serialize model:** `pickle`, `joblib`, ONNX for cross-platform.
- **Serve predictions:** REST API (FastAPI, Flask), batch inference, streaming (Kafka).
- **Monitor:** Track prediction drift, data drift, model performance over time.
- **Retrain triggers:** Set thresholds on performance degradation or data drift metrics.
- **MLOps tools:** MLflow, BentoML, SageMaker, Vertex AI, Seldon.

---

*End of ML Algorithms Handbook — Version 1.0*

> **Tip:** Use this document as a quick reference. For any algorithm, experiment on your actual data — empirical results always trump theoretical recommendations.
