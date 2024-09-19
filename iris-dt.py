import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")    #step 08 --> point 10

iris = load_iris()
X = iris.data 
y = iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 1

#apply mlflow
mlflow.set_experiment('iris-dt')     #ye line mention krne se abb jo run hoga vo iris-dt exp me jayega
                                     #also, agar ye exp create nhi ho rakha toh mlflow iss iris-dt ko create b kr dega 
                                     #but humne create kr rakha h ...

with mlflow.start_run():             #start_run(experiment_id = '101720669357673280') ... ye mention krne se b run iris-dt me jayega
                                     #yha maine iris-dt ki exp id copy ki h ... go to mlflow ui and vha se copy kr lo ... 
                                     #but hum upper wala method use kr kr rhe h ... 
    #iske ander hum jo b kaam krenge ... vo log hota jata h 
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    #abb hmme cheejo ko log krna h ... hum 2 cheejo ko log krenge ... 
    #1. hum apne parameters ko log krenge ... max_ddepth and n_estimators ... 
    #2. hum apni metric i.e. accuracy score ko log krenge ... 

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    

    #yhi hmara code ... 
    #with mlflow.start_run():  ---> isko hum context manager bolte h ... 
    #hum without context manager b kr skte the but aapko end_run() b krna padhta ... 
    #context manager k ander hum saara kaam kr rhe h toh end krne ki jrurat nhi h ... 
    #bcoz context manager automatically end kr deta h ...

    print('accuracy', accuracy) 



    #create a confusion matrix plot  #step 08 se related
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    #step 08 se related h ... 
    #save the plot as an artifact ... log_artifact() use hoga yha ...
    plt.savefig("confusion_matrix.png")
    #mlflow k liye code
    mlflow.log_artifact("confusion_matrix.png") 
    #log_artifact me aapko file ka path bta dena h ... upper savefig se meri file bann jayegi 
    #and thn m uss file path ko log_artifact me daal dunga ... 
    #run name hta deta hu m ... as pehle wale run k liye use ho chuka vo 

    mlflow.log_artifact(__file__)    #step 09 se related h 
    #logging of code ... code b artifact hi hota h ... hum artifact me iss current file ko hi
    #daal rhe h ... iss file ko as an artifact save kr rhe h hum ... 
    #so, artifact me hum file daal dete h ... jisko save krna chahte h ... 

    mlflow.sklearn.log_model(dt, 'decision tree')
   
   #mlflow ne scikit-learn k liye alag logging create kiya h ... tensorflow k liye alag logging 
   #create kiya h ... 
   #so, scikit learn k liye inka alag fn h ... i.e. log_model

   #hmare model ka name dt h and usko ek tag name diya h "decision tree"
   
   #note :- aap mlflow.log _model(dt, "decision tree") b use kr skte ho ...
            #but ye generic fn h ... jyada accha sklearn k sath rhega ... bcoz
            #sklearn k sath log_model or bhaut saara meta data utha k lata h 
            #plus ye b info aa jata h ki iss model ko serve kaise krna h ... 
   
    #step 11 --> adding tags
    mlflow.set_tag('author','anuj')
    mlflow.set_tag('model', 'decision tree')
    #yha aapko key value pair me tag btana hota h 
    #so, yha tags me hum kaisa b info daal skte h ... 
