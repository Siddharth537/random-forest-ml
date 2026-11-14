import pandas as pd
from pandas import read_csv 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
  
#load the dataset
data=read_csv(r'C:\Users\sheri\Downloads\diabetes (1).csv')
df=pd.DataFrame(data)

#split data into training and testing
x=df.drop(columns=['Outcome'])
y=df['Outcome'] 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#feauture scaling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


#converting to dataframe
x_train=pd.DataFrame(x_train_scaled,columns=x.columns)
x_test=pd.DataFrame(x_test_scaled,columns=x.columns)

#initialize and training
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)

#making predictions
y_pred=rf.predict(x_test)
#evaluate
accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy:{accuracy*100:.2f}%")

#confusion matrix
conf_matrix=confusion_matrix(y_test,y_pred)
print("confusion_matrix\n",conf_matrix)

#classification report
class_report=classification_report(y_test,y_pred)
print("classification_report\n",class_report)






import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, DetCurveDisplay
import numpy as np

# Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Feature Importance Bar Chart
feature_importances = rf.feature_importances_
features = x.columns
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(8,5))
sns.barplot(x=feature_importances[sorted_indices], y=[features[i] for i in sorted_indices], palette='viridis')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# ROC Curve
y_prob = rf.predict_proba(x_test)[:, 1]  # Get probability scores
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Detection Error Tradeoff (DET) Curve
plt.figure(figsize=(6,4))
DetCurveDisplay.from_predictions(y_test, y_prob)
plt.title('Detection Error Tradeoff (DET) Curve')
plt.show()
