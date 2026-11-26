# I Trained a Model on Random Labels. It Still Achieved 100% Training Accuracy.

*An experiment you can reproduce in Python that reveals the difference between memorization and learning, and why even powerful AI models can fail.*

Imagine training a machine learning model, watching its loss decrease, and finally seeing that beautiful number: **100% training accuracy.**

You're ready to put it on GitHub. Maybe even deploy it.

Then you evaluate it on unfamiliar data. Accuracy drops to 10.67%.

In a small experiment, two classifiers using the same algorithm and dataset produced remarkably different results:

- **Model A:** 100% training accuracy, 97.11% test accuracy.
- **Model B:** 100% training accuracy, 10.67% test accuracy.

The only thing that changed was the training labels. One could recognize unfamiliar handwritten digits. The other performed approximately as well as random guessing.

Both appeared perfect during training.

How do we know whether a machine learning model has discovered a useful pattern or simply remembered its training examples? And what does that question have to do with the neural networks and language models we build today?

Let's investigate using Python. No GPU, no massive dataset, and no advanced mathematical prerequisites.

## Abstract

This article examines the distinction between empirical fit and generalization through controlled experiments on handwritten digits, synthetic regression and feature representations. A one-nearest-neighbor classifier achieves perfect training accuracy with both genuine and permuted labels, while its held-out accuracy falls from 97.11% to 10.67% after permutation. We use this result to examine capacity, regularization, statistical estimation, validation and optimization, then connect classical learning theory to modern findings on memorization and double descent. The work is an educational investigation, not a claim of novel research. Its theoretical foundation is Chapter 5 of *Deep Learning* (Goodfellow et al., 2016).

**Keywords:** Generalization, statistical learning, memorization, regularization, representation learning.

## Table of contents

1. Experimental methodology: build a model that remembers everything
2. Experimental results: break the model without changing its architecture
3. What does it mean for a machine to learn?
4. Why model complexity matters
5. Stop accidentally cheating on your test set
6. The statistics hiding inside your loss function
7. Implement gradient descent yourself
8. Why deep learning needs useful representations
9. Why bigger neural networks sometimes generalize better
10. Your final challenge: break the model yourself
11. Discussion and limitations
12. Conclusion and references

## 1. Experimental methodology: build a model that remembers everything

We fix the stratified 75/25 split and label-permutation seed at 42. Both conditions use the same feature scaling and one-nearest-neighbor classifier. Only the training targets differ. Accuracy is measured against the supplied targets during training and the genuine targets during held-out evaluation. All hyperparameter selection uses training-set cross-validation.

We will use scikit-learn's handwritten digits dataset (scikit-learn developers, n.d.), which contains 1,797 grayscale images, each measuring 8 by 8 pixels. Our task is to identify the digit in an unfamiliar image.

Create a project and install the dependencies:

```bash
mkdir generalization-lab
cd generalization-lab
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scikit-learn
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

The input shape is `(1797, 64)`. Each row represents an image and each column a pixel intensity. The target array contains the corresponding digit labels.

Let's look at the data before training anything.

```python
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("digits.png", dpi=200)
plt.show()
```

*Figure 1. Handwritten digits from the dataset. Generated using the preceding code.*

We have reserved 450 images for testing and 1,347 for training. Stratification helps preserve the class proportions.

Our first algorithm is k-nearest neighbors (KNN). Unlike a neural network, it does not need gradient descent. It stores examples and predicts a label by looking at nearby training observations. With `k=1`, the closest observation decides the answer.

```python
def build_model(k=1):
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

clean_model = build_model().fit(X_train, y_train)
print("Train:", clean_model.score(X_train, y_train))
print("Test:", clean_model.score(X_test, y_test))
```

Recorded output:

```
Train: 1.0
Test:  0.9711111111111111
```

When evaluated on training examples, each observation is generally its own nearest neighbor. Provided there are no conflicting duplicate inputs, retrieving the correct label is trivial.

The interesting result is not the perfect training score. It is the 97.11% accuracy on unfamiliar images. That is evidence of generalization to this held-out sample.

## 2. Experimental results: break the model without changing its architecture

We are going to keep the images, preprocessing, and classifier exactly the same. But this time we will shuffle the training labels.

A picture of a seven might now be labeled three. A zero might be labeled eight. Shuffling preserves the overall label frequencies while disrupting their association with the images. Some labels will remain correct by chance.

Before running the next cell, predict what the training accuracy will be.

```python
rng = np.random.default_rng(42)
y_shuffled = rng.permutation(y_train)

random_model = build_model().fit(X_train, y_shuffled)
print("Train:", random_model.score(X_train, y_shuffled))
print("Test:", random_model.score(X_test, y_test))
```

Recorded output:

```
Train: 1.0
Test:  0.10666666666666667
```

There it is. Perfect training accuracy, despite having destroyed the meaningful relationship between the images and their labels.

```python
names = ["Correct labels", "Shuffled labels"]
train = [clean_model.score(X_train, y_train),
         random_model.score(X_train, y_shuffled)]
test = [clean_model.score(X_test, y_test),
        random_model.score(X_test, y_test)]
pos = np.arange(2)
plt.figure(figsize=(8, 5))
plt.bar(pos - .18, np.array(train)*100, .36, label="Training")
plt.bar(pos + .18, np.array(test)*100, .36, label="Testing")
plt.xticks(pos, names)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.legend()
plt.tight_layout()
plt.savefig("memorization.png", dpi=200)
plt.show()
```

*Figure 2. Both classifiers perfectly fit their respective training labels, but only one generalizes well to genuine labels.*

Notice that the training scores use the labels supplied to each model, whereas both test scores use genuine labels. This is an intentional experimental distinction.

Optimization and generalization are different questions. Optimization asks whether we can fit the observations. Generalization asks whether that fit helps with new observations.

### Does this happen with real neural networks?

In their ICLR 2017 paper, [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530), Chiyuan Zhang and colleagues (2017) demonstrated that convolutional neural networks trained with stochastic gradient methods could fit randomly assigned labels.

This challenged explanations of generalization based solely on limited representational capacity. A network capable of fitting arbitrary labels can also learn useful patterns when trained on meaningful data. Why does the training process favor useful solutions in some circumstances?

Our KNN demonstration is not a replication of their neural network experiments. It is a simpler way to see why fitting ability alone cannot explain generalization.

## 3. What does it mean for a machine to learn?

Mitchell (1997) describes learning in terms of a task, experience, and performance measure. In our example, the task is digit recognition, the experience is the training dataset, and the measure is accuracy on relevant examples.

Choosing that measure matters. A fraud classifier might achieve 99% accuracy by predicting that every transaction is legitimate if only 1% are fraudulent. A medical classifier might exploit differences between hospitals rather than disease features. A coding assistant might reproduce a familiar solution without handling a modified problem.

Goodfellow et al. (2016, Section 5.10) organize learning algorithms around four components:

- **Data:** the observations available to the system.
- **Model:** the family of relationships it can represent.
- **Objective:** the behavior rewarded during learning.
- **Optimization:** the procedure used to find a solution.

KNN is a special case without conventional iterative parameter optimization. Neural networks usually expose all four components explicitly.

Supervised learning uses targets, as in our digit classifier or a house-price regression model. Unsupervised learning discovers structure without those labels, as in clustering and PCA. Self-supervised learning constructs targets from the data itself, such as predicting the next token. Reinforcement learning instead involves an agent learning from interactions and rewards.

Generalization is commonly studied under the assumption that observations are independently drawn from the same distribution. Real systems can violate that assumption: users change, cameras differ, and historical data may no longer represent future inputs. Good performance on a random test split is not a guarantee of performance in production (Goodfellow et al., 2016; Geirhos et al., 2020).

## 4. Why model complexity matters

Imagine predicting the price of a house. A model that is too restrictive might miss important relationships. A model that is too flexible might fit every temporary fluctuation in the market.

Model capacity describes the family of functions a model can express. Effective capacity also depends on how it is trained and regularized (Goodfellow et al., 2016, Section 5.2).

Underfitting means the model fails to fit meaningful training patterns. Overfitting means its fit to training observations does not transfer sufficiently well to new ones.

Let's visualize this with a synthetic regression problem. Unlike a real dataset, we know the function generating these observations.

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
x = np.sort(rng.uniform(0, 1, 25)).reshape(-1, 1)
y_reg = np.sin(2*np.pi*x.ravel()) + rng.normal(0, .15, 25)
grid = np.linspace(0, 1, 500).reshape(-1, 1)
truth = np.sin(2*np.pi*grid.ravel())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for degree, ax in zip([1, 5, 15], axes):
    reg = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression()
    ).fit(x, y_reg)
    pred = reg.predict(grid)
    ax.scatter(x.ravel(), y_reg)
    ax.plot(grid.ravel(), truth, "--", label="True function")
    ax.plot(grid.ravel(), pred, label="Model")
    ax.set_title(f"Degree {degree}")
    ax.set_ylim(-2, 2)
    ax.legend()
    print(degree, mean_squared_error(truth, pred))
plt.tight_layout()
plt.savefig("polynomial_capacity.png", dpi=200)
plt.show()
```

*Figure 3. Polynomial regression at three levels of flexibility. The fixed vertical axis can clip extreme predictions; inspect the printed errors as well.*

The straight line cannot express the sine wave. A moderately flexible polynomial can approximate it. A high-degree polynomial can follow noise and become unstable between or beyond observations. In our run, the mean squared errors against the noise-free function were 0.1980, 0.00421, and 1115.67 for degrees 1, 5, and 15 respectively. The last result is heavily influenced by unstable extrapolation near the boundaries. Exact results depend on the sample and numerical conditioning.

### Bias, variance, and irreducible error

Bias measures systematic error in an estimator across repeated datasets. Variance measures how much its predictions change when the training sample changes (Bishop, 2006; Goodfellow et al., 2016).

For squared-error prediction, expected error can be decomposed into squared bias, variance, and irreducible noise. A restrictive model may have high bias; a sensitive model may have high variance. Even an ideal predictor cannot recover information absent from its inputs.

The classical tradeoff is useful, but it is not a universal law that test error must increase indefinitely with model size.

### Regularization: guide the model toward better solutions

Rather than removing all flexible functions from our hypothesis space, we can introduce a preference for smaller coefficients. Ridge regression does exactly that through an L2 penalty (Goodfellow et al., 2016).

```python
from sklearn.linear_model import Ridge

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for alpha, ax in zip([0.001, 0.1, 100], axes):
    reg = make_pipeline(
        PolynomialFeatures(15, include_bias=False),
        StandardScaler(), Ridge(alpha=alpha)
    ).fit(x, y_reg)
    ax.scatter(x.ravel(), y_reg)
    ax.plot(grid.ravel(), truth, "--")
    ax.plot(grid.ravel(), reg.predict(grid))
    ax.set_title(f"Alpha = {alpha}")
    ax.set_ylim(-2, 2)
plt.tight_layout()
plt.savefig("regularization.png", dpi=200)
plt.show()
```

*Figure 4. Changing regularization strength while keeping polynomial degree fixed.*

Too little regularization may permit unstable solutions; too much can produce underfitting. The right strength must be selected using validation data. Deep learning also uses dropout, augmentation, and early stopping. L2 penalties and decoupled weight decay are not generally equivalent with adaptive optimizers, as demonstrated by Loshchilov and Hutter (2019).

## 5. Stop accidentally cheating on your test set

A common mistake is trying many hyperparameters on a test set and reporting whichever produces the best result. The moment the test result influences your decision, it becomes part of model development.

Instead, choose configurations using validation data. We can use five-fold cross-validation within the training set: each fold becomes the validation set once, while the remaining four are used for fitting (Goodfellow et al., 2016, Section 5.3).

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    build_model(),
    {"kneighborsclassifier__n_neighbors": [1, 3, 5, 9, 15, 31, 61, 121]},
    cv=5, scoring="accuracy", return_train_score=True
)
search.fit(X_train, y_train)
print(search.best_params_)
print("CV accuracy:", search.best_score_)
print("Test accuracy:", search.score(X_test, y_test))
```

Recorded results: five neighbors achieved approximately 97.70% mean validation accuracy and 96.44% test accuracy.

Interestingly, the selected model scored slightly below our original one-neighbor classifier on the test set. Cross-validation provides an estimate, not a guarantee of ranking on every finite sample. The difference here is just three test images.

The existing lab includes the measured capacity plot and confusion matrix.

*Figure 5. Training and validation accuracy across different numbers of neighbors. Insert `figures/capacity.png` from the companion lab here.*

For a formal new study, reserve a fresh holdout if you have already used the original test scores to guide decisions. In this article the original holdout is used for a predefined educational comparison; do not tune against it.

Also avoid fitting preprocessing on the complete dataset before splitting. A pipeline fits its scaler within each training fold, preventing information from the validation fold from influencing the transformation. See scikit-learn developers (n.d.).

## 6. The statistics hiding inside your loss function

Suppose a language model predicts the next word in `The capital of France is ...`. If the observed continuation is Paris, we want the model to assign it high probability.

Maximum likelihood estimation seeks parameters that make observed data probable under a specified model. Using the logarithm converts products of likelihoods into sums. Minimizing negative log-likelihood for categorical targets gives the familiar cross-entropy objective (Goodfellow et al., 2016, Section 5.5).

```python
for p in [0.01, 0.10, 0.50, 0.90, 0.99]:
    print(f"Probability: {p:.2f}, loss: {-np.log(p):.4f}")
```

The loss drops from 4.6052 at 1% probability to 0.0101 at 99%. Confident mistakes receive a substantial penalty.

Mean squared error has a related interpretation: for independent Gaussian regression errors with fixed variance, minimizing squared error yields the same parameter optimum as maximizing likelihood. Laplace errors instead lead to an absolute-error objective.

Maximum likelihood can be consistent and asymptotically efficient under appropriate assumptions. It is not guaranteed to recover reality if the model family is misspecified or parameters are not identifiable.

### What if we have very little data?

Imagine observing eight heads and two tails. The maximum likelihood estimate of the probability of heads is 80%. A Bayesian approach combines a prior with the likelihood to obtain a posterior distribution (Bishop, 2006).

With a Beta(2,2) prior, our posterior is Beta(10,4).

```python
heads, tails = 8, 2
mle = heads / (heads + tails)
posterior_mean = (heads + 2) / (heads + tails + 4)
print(f"MLE: {mle:.2%}")
print(f"Posterior mean: {posterior_mean:.2%}")
```

The posterior mean is 71.43%. With 800 heads and 200 tails, it becomes approximately 79.88%. The relative influence of this prior decreases as evidence accumulates.

Full Bayesian prediction averages over posterior uncertainty rather than relying on a single parameter value. Maximum a posteriori (MAP) estimation instead selects a posterior mode. Under appropriate assumptions, a zero-centered Gaussian parameter prior yields an L2 penalty in the MAP objective.

## 7. Implement gradient descent yourself

Our classifier stores examples. A neural network usually needs to learn parameters. Let's implement a small regression training loop without an automatic differentiation framework.

```python
rng = np.random.default_rng(42)
x_gd = rng.uniform(-2, 2, 1000)
y_gd = 3*x_gd + 2 + rng.normal(0, .2, 1000)
w, b = 0.0, 0.0
lr = .05
history = []

for step in range(200):
    error = w*x_gd + b - y_gd
    history.append(np.mean(error**2))
    dw = 2*np.mean(error*x_gd)
    db = 2*np.mean(error)
    w -= lr*dw
    b -= lr*db

print("Weight:", w, "Bias:", b)
plt.plot(history)
plt.yscale("log")
plt.xlabel("Update")
plt.ylabel("Mean squared error")
plt.tight_layout()
plt.savefig("gradient_descent.png", dpi=200)
plt.show()
```

*Figure 6. The loss during full-batch gradient descent.*

The measured weight was 3.0101 and the intercept was 1.9845, close to the generating values of three and two. We never explicitly assigned those values to the model. We discovered them by measuring error and updating the parameters in the direction that reduces it.

For huge datasets, computing gradients from every observation at each step is expensive. Minibatch stochastic gradient descent estimates the gradient using a small random subset (Goodfellow et al., 2016, Section 5.9).

```python
w, b = 0.0, 0.0
for step in range(1000):
    idx = rng.choice(len(x_gd), 32, replace=False)
    xb, yb = x_gd[idx], y_gd[idx]
    error = w*xb + b - yb
    w -= lr*2*np.mean(error*xb)
    b -= lr*2*np.mean(error)
print(w, b)
```

For fixed model and batch sizes, an individual update does not directly scale with total dataset size. This is a foundation of scalable deep learning, although total training cost still depends on the number of updates and other factors.

## 8. Why deep learning needs useful representations

A 224 by 224 RGB image contains 150,528 numerical values. Two photographs of the same dog can differ dramatically at the pixel level because of lighting, background, and viewpoint.

KNN assumes nearby inputs tend to have related outputs. But raw pixel distance is not necessarily a good measure of semantic similarity.

The curse of dimensionality makes the problem harder. Dividing each dimension into ten regions creates ten regions in one dimension, a hundred in two, and ten billion in ten. A method that requires examples in every region quickly becomes impractical (Goodfellow et al., 2016, Section 5.11).

Let's sabotage our classifier by adding irrelevant features. We use a validation subset rather than repeatedly consulting the external test set.

```python
X_fit, X_valid, y_fit, y_valid = train_test_split(
    X_train, y_train, test_size=.25, random_state=42, stratify=y_train
)
rng = np.random.default_rng(42)
dimensions = [0, 10, 50, 100, 500, 1000]
scores = []
for d in dimensions:
    a = np.column_stack([X_fit, rng.normal(size=(len(X_fit), d))])
    v = np.column_stack([X_valid, rng.normal(size=(len(X_valid), d))])
    scores.append(build_model().fit(a, y_fit).score(v, y_valid))
plt.plot(dimensions, scores, "o-")
plt.xlabel("Irrelevant features")
plt.ylabel("Validation accuracy")
plt.tight_layout()
plt.savefig("noise_features.png", dpi=200)
plt.show()
```

*Figure 7. Measuring the effect of irrelevant dimensions on nearest-neighbor classification.*

As irrelevant coordinates increasingly dominate the distance, performance may deteriorate. More features are not necessarily more useful information.

### Compress the digits into two dimensions

Real-world data has structure. Rotating a mug changes thousands of pixels, but the underlying change might be controlled by one variable: rotation angle. This motivates the manifold hypothesis, the idea that meaningful high-dimensional observations may concentrate near lower-dimensional structures. It is a useful hypothesis, not a universal guarantee (Goodfellow et al., 2016).

Principal component analysis (PCA) learns linear directions of maximum variance without seeing class labels (Goodfellow et al., 2016, Section 5.8).

```python
from sklearn.decomposition import PCA

projection = make_pipeline(StandardScaler(), PCA(n_components=2))
z = projection.fit_transform(X_train)
plt.figure(figsize=(8, 6))
s = plt.scatter(z[:, 0], z[:, 1], c=y_train, cmap="tab10", s=12)
plt.colorbar(s, ticks=range(10))
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig("pca_digits.png", dpi=200)
plt.show()
```

*Figure 8. A two-dimensional projection of the handwritten digit training data.*

Some classes may form recognizable groups, while others overlap. PCA preserves directions of variance, not necessarily directions most useful for classification. K-means is another unsupervised approach that represents inputs by their proximity to learned cluster centers.

Deep networks can learn nonlinear, distributed representations. Convolutional layers can reuse spatial features, while transformers learn contextual token representations. Rather than learning a separate rule for every possible input, these systems can share information across examples through reusable features.

Other classical methods remain useful: logistic regression models class probabilities, decision trees partition the input space, and support vector machines can exploit kernels to learn nonlinear decision boundaries. Their suitability depends on the data and the task. Under the averaging assumptions of No Free Lunch results, no learner is universally superior (Wolpert, 1996). Useful learning depends on inductive biases that match real problems.

## 9. Why bigger neural networks sometimes generalize better

The classical picture of model complexity suggests that test error may rise when models become excessively flexible. But researchers have observed a more complicated pattern in some overparameterized systems.

In [Reconciling Modern Machine-Learning Practice and the Classical Bias-Variance Trade-Off](https://arxiv.org/abs/1812.11118), Belkin et al. (2019) described double descent. Test error can decrease, rise near the interpolation threshold, and then decrease again as capacity increases further.

Nakkiran et al. (2020) investigated this behavior in [Deep Double Descent](https://arxiv.org/abs/1912.02292), including neural network settings.

This does not mean larger models always generalize better or overfitting has disappeared. It means parameter count alone is insufficient to predict generalization. Architecture, data, optimization, and explicit or implicit regularization all matter.

The same distinction matters for language models. If a coding benchmark appears in a model's training corpus, a correct response is less informative about its ability to solve unfamiliar tasks. Carlini et al. (2021) demonstrated extraction of memorized training sequences from GPT-2 in [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805).

Memorization is not necessarily undesirable. Remembering facts and syntax is useful. The question is whether the model can apply information appropriately in new circumstances. Modified problems, contamination checks, and independent evaluations can provide additional evidence, although no single test proves understanding.

## 10. Your final challenge: break the model yourself

What if only some of our training labels are incorrect? Real datasets often contain annotation errors. Let's progressively corrupt the labels while retaining the original validation targets.

```python
def corrupt_labels(labels, fraction, seed=42):
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    count = int(len(labels)*fraction)
    idx = rng.choice(len(labels), count, replace=False)
    noisy[idx] = (noisy[idx] + rng.integers(1, 10, count)) % 10
    return noisy

fractions = [0, .1, .25, .5, .75, 1]
training, validation = [], []
for fraction in fractions:
    noisy = corrupt_labels(y_fit, fraction)
    classifier = build_model().fit(X_fit, noisy)
    training.append(classifier.score(X_fit, noisy))
    validation.append(classifier.score(X_valid, y_valid))
    print(fraction, training[-1], validation[-1])
plt.plot(np.array(fractions)*100, training, "o-", label="Training")
plt.plot(np.array(fractions)*100, validation, "o-", label="Validation")
plt.xlabel("Incorrect training labels (%)")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("label_corruption.png", dpi=200)
plt.show()
```

*Figure 9. Comparing training accuracy against corrupted targets with validation accuracy against genuine targets.*

In the recorded run, validation accuracy declined from 97.03% with clean labels to 68.84% with 25% corruption and 0.59% with full corruption, while training accuracy remained 100%. Unlike shuffling, this implementation guarantees every selected label changes. At full corruption, validation accuracy can even fall below chance because the correct class is deliberately excluded for each selected training example.

Try replacing KNN with logistic regression, a decision tree, or an MLP. Compare different amounts of data and different random seeds. Investigate which models fit random labels and which tolerate partial corruption.

### A debugging checklist for your next ML project

- Poor training and validation scores: investigate features, capacity, labels, and optimization. Try fitting a tiny subset first.
- Excellent training but poor validation: investigate overfitting and distribution differences.
- Excellent validation but poor production performance: investigate leakage, shortcut learning, and distribution shift.
- Suspiciously perfect scores: check duplicates, target leakage, and preprocessing before the split.
- A complicated model loses to a simple baseline: investigate why before increasing model size.

Use the metric that reflects your actual task. Accuracy is appropriate for this balanced digit dataset, but other applications may need precision, recall, calibration, or a cost-sensitive measure.

## 11. Discussion and limitations

The permutation experiment separates the ability to fit arbitrary labels from the ability to predict genuine labels. It does not establish that every interpolating model memorizes in the same way. KNN retains examples explicitly; neural networks encode information in learned parameters, and their behavior depends on the optimizer and architecture (Zhang et al., 2017).

A useful complementary explanation is shortcut learning: models may rely on predictive but non-robust cues that work within a benchmark yet fail when conditions change (Geirhos et al., 2020). Label corruption tests a different failure mode, so neither experiment substitutes for out-of-distribution evaluation.

The digits dataset is small, balanced and low-resolution. The reported accuracies describe one predefined stratified split, not a population-wide guarantee. The same holdout is used for the predefined baseline comparison and final selected model; it must not guide additional tuning. Cross-validation scores are dependent because training folds overlap, and their dispersion should not be interpreted as an independent-sample confidence interval. A stronger investigation would repeat the entire protocol over multiple seeds, use an external dataset, and report paired uncertainty estimates. The polynomial experiment also depends on numerical conditioning; its error against the known synthetic function is not equivalent to a real-world test error.

The results do not reproduce the large neural-network studies cited here. Double descent and language-model memorization are discussed as published findings, not as outcomes of our own KNN experiment.

## 12. Conclusion

We started with two models that achieved perfect training accuracy. One could recognize unfamiliar digits. The other remembered arbitrary labels.

The algorithm was the same. The information it received was different.

A model does not automatically know whether its labels are trustworthy, whether its correlations are meaningful, or whether its predictions will remain useful tomorrow. It operates on the data and assumptions we provide.

The next time you see training loss decrease, remember what you are actually observing: improvement on an objective. Whether that improvement transfers to the problem you care about is a separate question.

The most important question is not how well a model reproduces the past. It is what happens when the model encounters something it has never seen before.

## References

Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias-variance trade-off. *Proceedings of the National Academy of Sciences, 116*(32), 15849–15854. https://doi.org/10.1073/pnas.1903070116

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. https://link.springer.com/book/9780387310732

Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., & Raffel, C. (2021). Extracting training data from large language models. *30th USENIX Security Symposium*. https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*, 665–673. https://doi.org/10.1038/s42256-020-00257-z

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 5. https://www.deeplearningbook.org/contents/ml.html

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations*. https://arxiv.org/abs/1711.05101

Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.

Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2020). Deep double descent: Where bigger models and more data hurt. *International Conference on Learning Representations*. https://arxiv.org/abs/1912.02292

Scikit-learn developers. (n.d.). *Common pitfalls and recommended practices*. https://scikit-learn.org/stable/common_pitfalls.html

Scikit-learn developers. (n.d.). *load_digits*. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

Wolpert, D. H. (1996). The lack of a priori distinctions between learning algorithms. *Neural Computation, 8*(7), 1341–1390. https://doi.org/10.1162/neco.1996.8.7.1341

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *International Conference on Learning Representations*. https://arxiv.org/abs/1611.03530
