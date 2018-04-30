from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.decomposition import PCA


pipe_long = Pipeline([('scaler', MinMaxScaler()),
                      ('svm', SVC(C=100))])
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))

print(pipe_short.steps)


pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
pipe.fit(load_breast_cancer().data)

print(pipe.steps)
components = pipe.named_steps['pca'].components_
print('components shape: {}'.format(components.shape))
