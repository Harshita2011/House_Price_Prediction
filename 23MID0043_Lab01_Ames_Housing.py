"""Auto-generated from 23MID0043_Lab01_Ames_Housing.ipynb."""

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
SEED=42

# %%
# Download 'train.csv' from Kaggle House Prices
df=pd.read_csv('train.csv')
display(df.head())
print(df.shape)

# %%
display(df.info())
display(df.describe(include='all').T)
display(df.isnull().sum().sort_values(ascending=False).head(20))
print('Duplicates:',df.duplicated().sum())

# %%
sns.histplot(df['SalePrice'],kde=True)
plt.title('SalePrice Distribution')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.select_dtypes(include=np.number).corr(),cmap='coolwarm')
plt.show()

for col in ['GrLivArea','OverallQual','GarageCars','YearBuilt']:
    sns.scatterplot(data=df,x=col,y='SalePrice')
    plt.show()

sns.boxplot(data=df,x='OverallQual',y='SalePrice')
plt.xticks(rotation=45)
plt.show()

# %%
target='SalePrice'
drop_cols=['Id'] if 'Id' in df.columns else []
X=df.drop(columns=[target]+drop_cols)
y=df[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=SEED)

num=X.select_dtypes(include=np.number).columns
cat=X.select_dtypes(exclude=np.number).columns

# %%
numeric_pipe=Pipeline([
('imp',SimpleImputer(strategy='median')),
('scale',StandardScaler())
])

categorical_pipe=Pipeline([
('imp',SimpleImputer(strategy='most_frequent')),
('onehot',OneHotEncoder(handle_unknown='ignore'))
])

preprocess=ColumnTransformer([
('num',numeric_pipe,num),
('cat',categorical_pipe,cat)
])

# %%
dummy=DummyRegressor(strategy='mean')
dummy.fit(X_train[num],y_train)
pred=dummy.predict(X_test[num])
mae=mean_absolute_error(y_test,pred)
mse=mean_squared_error(y_test,pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,pred)
print('Baseline MAE:',mae)
print('Baseline MSE:',mse)
print('Baseline RMSE:',rmse)
print('Baseline R2:',r2)

# %%
simple=LinearRegression()
simple.fit(X_train[['GrLivArea']],y_train)
pred=simple.predict(X_test[['GrLivArea']])
print('R2',r2_score(y_test,pred))

# %%
models={
'Linear Regression':LinearRegression(),
'Ridge':Ridge(alpha=1.0),
'Lasso':Lasso(alpha=0.001,max_iter=20000),
'ElasticNet':ElasticNet(alpha=0.001,l1_ratio=0.5,max_iter=20000),
'Decision Tree':DecisionTreeRegressor(max_depth=6,random_state=SEED),
'Random Forest':RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=-1),
'Gradient Boosting':GradientBoostingRegressor(random_state=SEED)
}

rows=[]
fitted={}
for name,est in models.items():
    pipe=Pipeline([('preprocess',preprocess),('model',est)])
    pipe.fit(X_train,y_train)
    p=pipe.predict(X_test)
    mae=mean_absolute_error(y_test,p)
    mse=mean_squared_error(y_test,p)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,p)
    # Adjusted R2 using the number of features after preprocessing
    Xt=pipe.named_steps['preprocess'].transform(X_test)
    n,n_features=Xt.shape
    adj_r2=np.nan if n<=n_features+1 else 1-(1-r2)*(n-1)/(n-n_features-1)
    rows.append({
        'Model':name,
        'MAE':mae,
        'MSE':mse,
        'RMSE':rmse,
        'R2':r2,
        'Adjusted_R2':adj_r2
    })
    fitted[name]=pipe

results_df=pd.DataFrame(rows).sort_values('RMSE')
display(results_df)

# %%
poly=Pipeline([
('imp',SimpleImputer(strategy='median')),
('poly',PolynomialFeatures(degree=2,include_bias=False)),
('scale',StandardScaler()),
('model',Ridge(alpha=1.0))
])
features=['GrLivArea','OverallQual','YearBuilt']
poly.fit(X_train[features],y_train)
print(poly.score(X_test[features],y_test))

# %%
cv=KFold(n_splits=5,shuffle=True,random_state=SEED)
cv_rows=[]
for name,est in models.items():
    pipe=Pipeline([('preprocess',preprocess),('model',est)])
    scores=cross_validate(pipe,X_train,y_train,cv=cv,
                          scoring=['neg_root_mean_squared_error','r2'])
    cv_rows.append([name,
                    -scores['test_neg_root_mean_squared_error'].mean(),
                    scores['test_r2'].mean()])
cv_df=pd.DataFrame(cv_rows,columns=['Model','CV_RMSE','CV_R2'])
display(cv_df)

# %%
grid=GridSearchCV(
Pipeline([('preprocess',preprocess),('model',Ridge())]),
{'model__alpha':[0.001,0.01,0.1,1,10,100]},
cv=5,
scoring='neg_root_mean_squared_error')
grid.fit(X_train,y_train)
print(grid.best_params_)
print(-grid.best_score_)

# %%
best=fitted[results_df.iloc[0]['Model']]
pred=best.predict(X_test)
res=y_test-pred

plt.scatter(pred,res)
plt.axhline(0,color='red')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.show()

sns.histplot(res,kde=True)
plt.show()

plt.scatter(y_test,pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# %%
model=best.named_steps['model']
if hasattr(model,'feature_importances_'):
    imp=pd.DataFrame({
        'Feature':best.named_steps['preprocess'].get_feature_names_out(),
        'Importance':model.feature_importances_
    }).sort_values('Importance',ascending=False)
    display(imp.head(20))

# %%
results_df.to_csv('ames_model_comparison.csv',index=False)
cv_df.to_csv('ames_cross_validation.csv',index=False)
joblib.dump(best,'ames_house_price_pipeline.joblib')
