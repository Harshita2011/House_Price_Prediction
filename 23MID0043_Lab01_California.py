"""Auto-generated from 23MID0043_Lab01_California.ipynb."""

# %%
import warnings; warnings.filterwarnings('ignore')
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
print(platform.python_version())

# %%
housing=fetch_california_housing(as_frame=True)
df=housing.frame.rename(columns={'MedHouseVal':'Price'})
display(df.head())

# %%
display(df.shape)
display(df.info())
display(df.describe().T)
display(df.isna().sum())
display(df.duplicated().sum())

# %%
sns.histplot(df['Price'],kde=True); plt.show()
plt.figure(figsize=(10,8)); sns.heatmap(df.corr(),cmap='coolwarm'); plt.show()
for col in ['MedInc','HouseAge','AveRooms','AveBedrms','Population']:
    sns.scatterplot(data=df,x=col,y='Price'); plt.show()
for c in df.columns:
    sns.boxplot(x=df[c]); plt.title(c); plt.show()

# %%
X=df.drop(columns='Price'); y=df['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=SEED)

# %%
num=X.select_dtypes(include=np.number).columns
cat=X.select_dtypes(exclude=np.number).columns
pre=ColumnTransformer([
('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)
])

# %%
d=DummyRegressor()
d.fit(X_train[num],y_train)
p=d.predict(X_test[num])
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
slr.fit(X_train[['MedInc']],y_train)
pred=slr.predict(X_test[['MedInc']])
print(r2_score(y_test,pred))

# %%
models={
'Linear Regression':LinearRegression(),
'Ridge':Ridge(),
'Lasso':Lasso(),
'ElasticNet':ElasticNet(),
'Decision Tree':DecisionTreeRegressor(random_state=SEED),
'Random Forest':RandomForestRegressor(random_state=SEED),
'Gradient Boosting':GradientBoostingRegressor(random_state=SEED)}
results=[]; fitted={}
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
 fitted[n]=pipe
results_df=pd.DataFrame(results,columns=['Model','MAE','MSE','RMSE','R2','Adjusted_R2']).sort_values('RMSE')
display(results_df)

# %%
poly=Pipeline([
('imp',SimpleImputer(strategy='median')),
('poly',PolynomialFeatures(degree=2,include_bias=False)),
('scale',StandardScaler()),
('model',Ridge(alpha=1.0))
])
poly.fit(X_train[['MedInc','HouseAge','AveRooms']],y_train)
print(poly.score(X_test[['MedInc','HouseAge','AveRooms']],y_test))

# %%
cv=KFold(n_splits=5,shuffle=True,random_state=SEED)
rows=[]
for n,m in models.items():
 pipe=Pipeline([('pre',pre),('model',m)])
 s=cross_validate(pipe,X_train,y_train,cv=cv,scoring=['neg_root_mean_squared_error','r2'])
 rows.append([n,-s['test_neg_root_mean_squared_error'].mean(),s['test_r2'].mean()])
cv_df=pd.DataFrame(rows,columns=['Model','CV_RMSE','CV_R2'])
display(cv_df)

# %%
grid=GridSearchCV(Pipeline([('pre',pre),('model',Ridge())]),
{'model__alpha':[0.001,0.01,0.1,1,10,100]},cv=5,
scoring='neg_root_mean_squared_error')
grid.fit(X_train,y_train)
print(grid.best_params_, -grid.best_score_)

# %%
best=fitted[results_df.iloc[0]['Model']]
pred=best.predict(X_test)
res=y_test-pred
plt.scatter(pred,res); plt.axhline(0,color='r'); plt.show()
sns.histplot(res,kde=True); plt.show()
plt.scatter(y_test,pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
plt.show()

# %%
if hasattr(best.named_steps['model'],'feature_importances_'):
    fi=pd.DataFrame({'Feature':best.named_steps['pre'].get_feature_names_out(),
                     'Importance':best.named_steps['model'].feature_importances_}).sort_values('Importance',ascending=False)
    display(fi.head(20))
elif hasattr(best.named_steps['model'],'coef_'):
    coef=pd.DataFrame({'Feature':best.named_steps['pre'].get_feature_names_out(),
                       'Coefficient':best.named_steps['model'].coef_})
    display(coef.head())

# %%
results_df.to_csv('California_model_comparison.csv',index=False)
cv_df.to_csv('California_cross_validation_results.csv',index=False)
joblib.dump(best,'California_house_price_pipeline.joblib')
