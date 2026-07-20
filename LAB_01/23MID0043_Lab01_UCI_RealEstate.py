"""Auto-generated from 23MID0043_Lab01_UCI_RealEstate.ipynb."""

# %%
# Install once if needed
!pip install -q ucimlrepo joblib

# %%
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, platform, joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split,KFold,cross_validate,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
SEED=42

# %%
import sklearn

np.random.seed(SEED)

print('Python:', platform.python_version())
print('pandas:', pd.__version__)
print('numpy:', np.__version__)
print('scikit-learn:', sklearn.__version__)

# %%
from ucimlrepo import fetch_ucirepo


real_estate=fetch_ucirepo(id=477)
X=real_estate.data.features
y=real_estate.data.targets.iloc[:,0]
df=X.copy()
df['HousePrice']=y
display(df.head())
print(df.shape)

# %%
df.info()

# %%
df.describe().T

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
df.dtypes

# %%
sns.histplot(df['HousePrice'],kde=True);plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.show()

for col in df.columns:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col]);plt.title(col);plt.show()

for col in X.columns:
    plt.figure(figsize=(5,3))
    sns.scatterplot(data=df,x=col,y='HousePrice')
    plt.show()

sns.pairplot(df)
plt.show()

# %%
num=X.columns
pre=ColumnTransformer([
('num',Pipeline([
('imp',SimpleImputer(strategy='median')),
('scale',StandardScaler())
]),num)
])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# %%
from sklearn.dummy import DummyRegressor
dummy=DummyRegressor()
dummy.fit(X_train,y_train)
p=dummy.predict(X_test)
mae=mean_absolute_error(y_test,p)
mse=mean_squared_error(y_test,p)
rmse=np.sqrt(mse)
r2=r2_score(y_test,p)
print('Baseline MAE:',mae)
print('Baseline MSE:',mse)
print('Baseline RMSE:',rmse)
print('Baseline R2:',r2)

# %%
slr=LinearRegression()
slr.fit(X_train[['X2 house age']],y_train)
pred=slr.predict(X_test[['X2 house age']])
print('R2',r2_score(y_test,pred))

# %%
models={
'Linear Regression':LinearRegression(),
'Ridge':Ridge(),
'Lasso':Lasso(alpha=0.001,max_iter=20000),
'ElasticNet':ElasticNet(alpha=0.001,max_iter=20000),
'Decision Tree':DecisionTreeRegressor(random_state=42),
'Random Forest':RandomForestRegressor(n_estimators=300,random_state=42),
'Gradient Boosting':GradientBoostingRegressor(random_state=42)
}

results=[]
pipes={}
for n,m in models.items():
    pipe=Pipeline([('pre',pre),('model',m)])
    pipe.fit(X_train,y_train)
    pr=pipe.predict(X_test)
    mae=mean_absolute_error(y_test,pr)
    mse=mean_squared_error(y_test,pr)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,pr)
    Xt=pipe.named_steps['pre'].transform(X_test)
    n_samples,n_features=Xt.shape
    adj_r2=np.nan if n_samples<=n_features+1 else 1-(1-r2)*(n_samples-1)/(n_samples-n_features-1)
    results.append([n,mae,mse,rmse,r2,adj_r2])
    pipes[n]=pipe

results_df=pd.DataFrame(results,columns=['Model','MAE','MSE','RMSE','R2','Adjusted_R2']).sort_values('RMSE')
display(results_df)

# %%
poly=Pipeline([
('imp',SimpleImputer(strategy='median')),
('poly',PolynomialFeatures(degree=2,include_bias=False)),
('scale',StandardScaler()),
('model',Ridge())
])
feat=X.columns[:3]
poly.fit(X_train[feat],y_train)
print(poly.score(X_test[feat],y_test))

# %%
cv=KFold(n_splits=5,shuffle=True,random_state=42)
rows=[]
for n,m in models.items():
    pipe=Pipeline([('pre',pre),('model',m)])
    sc=cross_validate(pipe,X_train,y_train,cv=cv,
                      scoring=['neg_root_mean_squared_error','r2'])
    rows.append([n,-sc['test_neg_root_mean_squared_error'].mean(),sc['test_r2'].mean()])
cv_df=pd.DataFrame(rows,columns=['Model','CV_RMSE','CV_R2'])
display(cv_df)

# %%
grid=GridSearchCV(
Pipeline([('pre',pre),('model',Ridge())]),
{'model__alpha':[0.001,0.01,0.1,1,10,100]},
cv=5,
scoring='neg_root_mean_squared_error')
grid.fit(X_train,y_train)
print(grid.best_params_)
print(-grid.best_score_)

# %%
best=pipes[results_df.iloc[0]['Model']]
pred=best.predict(X_test)
res=y_test-pred

plt.scatter(pred,res)
plt.axhline(0,color='red')
plt.show()

sns.histplot(res,kde=True)
plt.show()

plt.scatter(y_test,pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
plt.show()

# %%
m=best.named_steps['model']
if hasattr(m,'feature_importances_'):
    fi=pd.DataFrame({
        'Feature':best.named_steps['pre'].get_feature_names_out(),
        'Importance':m.feature_importances_
    }).sort_values('Importance',ascending=False)
    display(fi)
elif hasattr(m,'coef_'):
    coef=pd.DataFrame({
        'Feature':best.named_steps['pre'].get_feature_names_out(),
        'Coefficient':m.coef_
    })
    display(coef)

# %%
results_df.to_csv('uci_model_comparison.csv',index=False)
cv_df.to_csv('uci_cross_validation.csv',index=False)
joblib.dump(best,'uci_real_estate_model.joblib')
