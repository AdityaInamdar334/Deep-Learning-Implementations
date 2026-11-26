import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
out=Path(__file__).resolve().parent/'outputs'; out.mkdir(exist_ok=True)
X,y=load_digits(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
def model(k): return make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=k))
for name, labels in [('real',y_train),('shuffled',np.random.default_rng(42).permutation(y_train))]:
    m=model(1).fit(X_train,labels)
    print(name, 'train',accuracy_score(labels,m.predict(X_train)), 'test',m.score(X_test,y_test))
search=GridSearchCV(model(1),{'kneighborsclassifier__n_neighbors':[1,3,5,9,15,31,61,121]},cv=5,scoring='accuracy',return_train_score=True)
search.fit(X_train,y_train)
r=search.cv_results_; ks=[p['kneighborsclassifier__n_neighbors'] for p in r['params']]
print('CV scores:',list(zip(ks,r['mean_test_score'].round(4))))
print('best k',search.best_params_,'test',search.score(X_test,y_test))
plt.figure(); plt.plot(ks,r['mean_train_score'], 'o-',label='Training'); plt.plot(ks,r['mean_test_score'],'o-',label='Validation'); plt.xscale('log'); plt.xlabel('Neighbors (k)'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout(); plt.savefig(out/'capacity.png',dpi=180); plt.close()
ConfusionMatrixDisplay(confusion_matrix(y_test,search.predict(X_test))).plot(); plt.tight_layout(); plt.savefig(out/'confusion.png',dpi=180); plt.close()
