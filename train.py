import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
ordinal=OrdinalEncoder()

##Read file and Ordinal encoding
df=pd.read_csv(r'./gender_classification_v7.csv')
df['gender']=ordinal.fit_transform(df['gender'].values.reshape(-1,1))

##Random forest and Train-Test
rfr=RandomForestClassifier(n_estimators=150,max_depth=4)
x_train,x_test,y_train,y_test=train_test_split(df.drop('gender',axis=1),df.gender,test_size=0.25,random_state=42)

##Model train
rfr.fit(x_train,y_train)
y_pred=rfr.predict(x_test)
print(y_pred)

##Model eval
acc=accuracy_score(y_test,y_pred)
with open('metrics.txt','w') as outfile:
    outfile.write("Test data accuracy is : %2.6f%%\n" %acc*10)
ax_con=sb.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='Blues',fmt="f")
plt.tight_layout()
ax_con.set_xlabel('True/False')
ax_con.set_ylabel('True/False')
ax_con.set_title('Confusion Matrix')
plt.savefig("Confusion_matrix.png",dpi=120) 
plt.close()

##Feature Importance
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
imp=rfr.feature_importances_
labels=df.columns

imp_df=pd.DataFrame(list(zip(imp,labels)),columns=['Importance','Names'])
ax=sb.barplot(x='Importance',y='Names',data=imp_df)
ax.set_xlabel('Importance') 
ax.set_ylabel('Feature')#ylabel
ax.set_title('Random forest\nfeature importance')
plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()

##Residula Plots
y_pred = rfr.predict(x_test) + np.random.normal(0,0.25,len(y_test))
y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])
print(res_df.head())
ax = sb.scatterplot(x="true", y="pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True gender',fontsize = axis_fs) 
ax.set_ylabel('Predicted gender', fontsize = axis_fs)#ylabel
ax.set_title('Residuals', fontsize = title_fs)

plt.tight_layout()
plt.savefig("residuals.png",dpi=120) 