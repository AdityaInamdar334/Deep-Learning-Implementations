import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent / "figures")
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


fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("digits.png", dpi=200)
plt.close("all")


def build_model(k=1):
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

clean_model = build_model().fit(X_train, y_train)
print("Train:", clean_model.score(X_train, y_train))
print("Test:", clean_model.score(X_test, y_test))


rng = np.random.default_rng(42)
y_shuffled = rng.permutation(y_train)

random_model = build_model().fit(X_train, y_shuffled)
print("Train:", random_model.score(X_train, y_shuffled))
print("Test:", random_model.score(X_test, y_test))


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
plt.close("all")


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
plt.close("all")


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
plt.close("all")


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


for p in [0.01, 0.10, 0.50, 0.90, 0.99]:
    print(f"Probability: {p:.2f}, loss: {-np.log(p):.4f}")


heads, tails = 8, 2
mle = heads / (heads + tails)
posterior_mean = (heads + 2) / (heads + tails + 4)
print(f"MLE: {mle:.2%}")
print(f"Posterior mean: {posterior_mean:.2%}")


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
plt.close("all")


w, b = 0.0, 0.0
for step in range(1000):
    idx = rng.choice(len(x_gd), 32, replace=False)
    xb, yb = x_gd[idx], y_gd[idx]
    error = w*xb + b - yb
    w -= lr*2*np.mean(error*xb)
    b -= lr*2*np.mean(error)
print(w, b)


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
plt.close("all")


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
plt.close("all")


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
plt.close("all")
