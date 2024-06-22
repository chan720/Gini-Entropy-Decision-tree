import pandas 
from sklearn import datasets 
from sklearn import tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.model_selection import train_test_split
# df=pandas.read_csv("zoo.csv")
# feature=['animal_name','hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','class_type']
iris=datasets.load_iris()
x=iris.data
y=iris.target
# print(x.shape)
# print(y.shape)
train_feature=x[:80,:-1]
test_feature=x[80:,:-1]
train_target=y[:80]
test_target=y[80:]
tree=DecisionTreeClassifier(criterion='entropy')
tree.fit(train_feature,train_target)
prediction=tree.predict(test_feature)
print("By manually data")
print("=============ENTROPY================")
print("prediction: ",prediction[:7])
print("Accuracy on test data: ",tree.score(test_feature,test_target)*100, "%")
print("------------------------------------------------")
print("Accuracy on train data: ",tree.score(train_feature,train_target)*100, "%")

# Code for GINI
print("=============GINI ================")
ginitree=DecisionTreeClassifier(criterion='gini')
ginitree.fit(train_feature,train_target)
prediction=ginitree.predict(test_feature)
print("prediction: ",prediction[:7])
print("Accuracy on test data: ",tree.score(test_feature,test_target)*100, "%")
print("------------------------------------------------")
print("Accuracy on train data: ",tree.score(train_feature,train_target)*100, "%")
print("\n\n======By random data========")
print("=============ENTROPY================")

# Random

iris = datasets.load_iris()
x = iris.data
y = iris.target
train_feature, test_feature, train_target, test_target = train_test_split(x, y, test_size=0.2, random_state=42)
tree = DecisionTreeClassifier(criterion='entropy')
ginitree = DecisionTreeClassifier(criterion='gini')
tree.fit(train_feature, train_target)
prediction_entropy = tree.predict(test_feature)
print("Entropy Tree Predictions: ", prediction_entropy[:7])
print("Accuracy on test data: ", tree.score(test_feature, test_target) * 100, "%")
print("Accuracy on training data: ", tree.score(train_feature, train_target) * 100, "%")
ginitree.fit(train_feature, train_target)
prediction_gini = ginitree.predict(test_feature)
print("============= GINI ================")
print("Gini Tree Predictions: ", prediction_gini[:7])
print("Accuracy on test data: ", ginitree.score(test_feature, test_target) * 100, "%")
print("Accuracy on training data: ", ginitree.score(train_feature, train_target) * 100, "%")

ginitree.fit(train_feature, train_target)
ginitree_rules = export_text(ginitree, feature_names=iris.feature_names)
print("\nDecision Tree (Gini) Rules:")
print(ginitree_rules)
