import pandas as pd
test="C:\\Users\\izuma/test.tsv"
train="C:\\Users\\izuma/train.tsv"
test_data=pd.read_csv(test,sep="\t")
train_data=pd.read_csv(train,sep="\t")

test_data["sex"]=test_data["sex"].apply(lambda x: 1 if x=="female" else 0)
train_data["sex"]=train_data["sex"].apply(lambda x: 1 if x=="female" else 0)

#データを絞り込みsurvivedは目的変数→train_yへ
train_X=train_data.drop(["id","survived","age","embarked"],axis=1)#axis=0行、axis=1列  dropしたもの以外を入れる
train_y=train_data["survived"]

val_X=test_data.drop(["id","age","embarked"],axis=1)
from sklearn.ensemble import GradientBoostingClassifier as GB
model = GB(random_state=0,learning_rate=0.01)
model.fit(train_X,train_y)
val_predictions=model.predict(val_X)
print(model.score(train_X,train_y))
from sklearn.ensemble import RandomForestClassifier as RF
random_model = RF(n_estimators=1000,random_state=0)
random_model.fit(train_X,train_y)
val_predictions=random_model.predict(val_X)
print(random_model.score(train_X,train_y))
print(val_predictions)
#val_predictions_int=val_predictions.astype('int64')
test_data["survived"]=val_predictions
test=test_data[["id","survived"]]
test.to_csv("sample_submit.tsv",index=False,header=False,encoding='cp932',sep="\t")