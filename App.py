import streamlit as st
import pandas as pd
import pickle
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split


ds = pd.read_csv('r''https://raw.githubusercontent.com/SYEDABUTHAHIRM102/Diabetes-Prediction/main/Data/diabetes%20dataset.csv')
dtree_model = pickle.load(open('D:\project\Decision Tree.sav', 'rb'))
knn_model = pickle.load(open('D:\project\KNN.sav', 'rb'))
lsvm_model = pickle.load(open('D:\project\Linear SVM.sav', 'rb'))
lr_model = pickle.load(open('D:\project\Logistic Regression.sav', 'rb'))
nb_model = pickle.load(open('D:\project\_Naive Bayes.sav', 'rb'))
rfor_model = pickle.load(open('D:\project\Random Forest.sav', 'rb'))

#X = ds.drop(['Result'], axis= 1)
#Y = ds.iloc[:, -1]

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0)

#new_title = '<p style="font-family:sans-serif; color:orange; font-size: 20px;">After 10 Years</p>'

with st.sidebar:
    selectbox = st.selectbox("**ALGORITHMS**", 
                             options=("Home", 
                                      "Logistic Regression",
                                      "K-Nearest Neighbor", 
                                      "Naive Bayes", 
                                      "Support Vector Machine", 
                                      "Random Forest", 
                                      "Decision Tree", 
                                      "All Algorithms")
                                      )
      
    st.write()

if(selectbox == 'Home'):

    st.sidebar.info('''*Machine Learning algorithms are the programs that can learn the hidden patterns from the data, predict the output, and improve the performance from experiences on their own. Different algorithms can be used in machine learning for different tasks.*''')
    st.sidebar.info('''*Machine Learning Algorithm can be broadly classified into three types*''')
    st.sidebar.info('''*1 Supervised Learning*''')
    st.sidebar.info('''*Supervised learning is a type of Machine learning in which the machine needs external supervision to learn. The supervised learning models are trained using the labeled dataset. Once the training and processing are done, the model is tested by providing a sample test data to check whether it predicts the correct output.*''')
    st.sidebar.info('''*Supervised learning can be divided further into four categories of problem*''')
    st.sidebar.info('''*1 (a) Regression*''')
    st.sidebar.info('''*i Linear*''')
    st.sidebar.info('''*ii Polynomial*''')
    st.sidebar.info('''*1 (b) Decision Tree*''')
    st.sidebar.info('''*1 (c) Random Forest*''')
    st.sidebar.info('''*1 (d) Classification*''')
    st.sidebar.info('''*i KNN*''')
    st.sidebar.info('''*ii Tress*''')
    st.sidebar.info('''*iii Logistic Regression*''')
    st.sidebar.info('''*iv Naive Bayes*''')
    st.sidebar.info('''*v SVM*''')
    st.sidebar.info('''*2 Unsupervised Learning*''')
    st.sidebar.info('''*It is a type of machine learning in which the machine does not need any external supervision to learn from the data, hence called unsupervised learning. The unsupervised models can be trained using the unlabelled dataset that is not classified, nor categorized, and the algorithm needs to act on that data without any supervision. In unsupervised learning, the model doesn't have a predefined output, and it tries to find useful insights from the huge amount of data. These are used to solve the Association and Clustering problems.*''')
    st.sidebar.info('''*3 Reinforcement Learning*''')
    st.sidebar.info('''*In Reinforcement learning, an agent interacts with its environment by producing actions, and learn with the help of feedback. The feedback is given to the agent in the form of rewards, such as for each good action, he gets a positive reward, and for each bad action, he gets a negative reward. There is no supervision provided to the agent. Q-Learning algorithm is used in reinforcement learning.*''') 

    st.title("DIABETES PREDICTION")
    st.subheader("Using Machine Learning")

    st.markdown('*Diabetes*')
    st.markdown('Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose. Type 1 and type 2 diabetes are the most common forms of the disease, but there are also other kinds, such as gestational diabetes, which occurs during pregnancy, as well as other forms.')
    st.markdown('*Machine Learning*')
    st.markdown('Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.')

    st.subheader('About(Project)')
    st.markdown(' The aim of this project is to develop a system which can perform early prediction of diabetes for a patient with a higher accuracy by combining the results of different machine learning techniques. The algorithms like Naive Bayes, K-Nearest Neighbor, Logistic Regression, Random Forest, Support Vector Machine and Decision Tree are used.')

    st.subheader('Data(Training)')
    st.write(ds.describe())

    st.subheader('Visualization')
    st.bar_chart(ds)


if(selectbox == 'Logistic Regression'):

    st.sidebar.info('''*Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.*''')
    st.sidebar.info('''*Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.*''')
    st.sidebar.info('''*Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems.*''')

    st.title('LOGISTIC REGRESSION')
    
    #lr_model.fit(X_train, Y_train)

    st.markdown('*Accuracy Score*')
    
    #acc = (str(accuracy_score(Y_test, lr_model.predict(X_test))*100)+'%')
    st.warning('0.780')
 
    col1, col2, = st.columns(2)
    
    with col1:
      Pregnancies = st.number_input('Number of Pregnancies')
    with col2:
      Glucose = st.number_input('Glucose Level')
    with col1:
      BP = st.number_input('Blood Pressure value')
    with col2:
      SkinThickness = st.number_input('Skin Thickness value')
    with col1:
      Insulin = st.number_input('Insulin Level')
    with col2:
      BMI = st.number_input('BMI value')
    with col1:
      DPF = st.number_input('Diabetes Pedigree Function value')
    with col2:
      Age = st.number_input('Age of the Person')

    #Code for prediction
    
    Age1=Age+10

    diabetes_diagnosis0 = ''
    diabetes_diagnosis1 = ''

    diab_prediction = lr_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
    diab_prediction1 = lr_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])

    #Creating a button for prediction
  
    if st.button('Diabetes Test Result'):
      
      if (diab_prediction[0]==0):
        diabetes_diagnosis0 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis0)
             
        if(diab_prediction1[0]==0):          
          diabetes_diagnosis1= 'Not Possible'
             
        else:
          diabetes_diagnosis1='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.subheader('After 10 Years') 
        st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis1)
      else:
        diabetes_diagnosis0='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis0)       

   


if(selectbox == 'K-Nearest Neighbor'):
    st.sidebar.info('''*K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.KNN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.KNN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using KNN algorithm.*''')
    st.sidebar.info('''*KNN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.KNN is a non-parametric algorithm, which means it does not make any assumption on underlying data.*''')
    st.sidebar.info('''*It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.*''')

    st.title('K-NEAREST NEIGHBOR')

    #knn_model.fit(X_train, Y_train)

    st.markdown('*Accuracy Score*')
    
    #acc1 = (str(accuracy_score(Y_test, knn_model.predict(X_test))*100)+'%')
    st.warning('0.873')

    col1, col2, = st.columns(2)
   
    with col1:
      Pregnancies = st.number_input('Number of Pregnancies')
    with col2:
      Glucose = st.number_input('Glucose Level')
    with col1:
      BP = st.number_input('Blood Pressure value')
    with col2:
      SkinThickness = st.number_input('Skin Thickness value')
    with col1:
      Insulin = st.number_input('Insulin Level')
    with col2:
      BMI = st.number_input('BMI value')
    with col1:
      DPF = st.number_input('Diabetes Pedigree Function value')
    with col2:
      Age = st.number_input('Age of the Person')

    #Code for prediction
    
    Age1=Age+10

    diabetes_diagnosis0 = ''
    diabetes_diagnosis1 = ''

    diab_prediction = knn_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
    diab_prediction1 = knn_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])

    #Creating a button for prediction
  
    if st.button('Diabetes Test Result'):
      
      if (diab_prediction[0]==0):
        diabetes_diagnosis0 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis0)
             
        if(diab_prediction1[0]==0):          
          diabetes_diagnosis1= 'Not Possible'
             
        else:
          diabetes_diagnosis1='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.subheader('After 10 Years') 
        st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis1)
      else:
        diabetes_diagnosis0='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis0)

    


if(selectbox == 'Naive Bayes'):
    st.sidebar.info('''*Naive Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.It is mainly used in text classification that includes a high-dimensional training dataset.*''')
    st.sidebar.info('''*Naive Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.*''')
    st.sidebar.info('''*It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.Some popular examples of Naive Bayes Algorithm are spam filtration, Sentimental analysis, and classifying articles.*''')

    st.title('NAIVE BAYES')

    #nb_model.fit(X_train, Y_train)

    st.markdown('*Accuracy Score*')
    
    #acc2 = (str(accuracy_score(Y_test, nb_model.predict(X_test))*100)+'%')
    st.warning(0.763)

    col1, col2, = st.columns(2)
   
    with col1:
      Pregnancies = st.number_input('Number of Pregnancies')
    with col2:
      Glucose = st.number_input('Glucose Level')
    with col1:
      BP = st.number_input('Blood Pressure value')
    with col2:
      SkinThickness = st.number_input('Skin Thickness value')
    with col1:
      Insulin = st.number_input('Insulin Level')
    with col2:
      BMI = st.number_input('BMI value')
    with col1:
      DPF = st.number_input('Diabetes Pedigree Function value')
    with col2:
      Age = st.number_input('Age of the Person')
    
    #Code for prediction
    
    Age1=Age+10

    diabetes_diagnosis0 = ''
    diabetes_diagnosis1 = ''

    diab_prediction = nb_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
    diab_prediction1 = nb_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])

    #Creating a button for prediction
  
    if st.button('Diabetes Test Result'):
      
      if (diab_prediction[0]==0):
        diabetes_diagnosis0 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis0)
             
        if(diab_prediction1[0]==0):          
          diabetes_diagnosis1= 'Not Possible'
             
        else:
          diabetes_diagnosis1='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.subheader('After 10 Years') 
        st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis1)
      else:
        diabetes_diagnosis0='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis0)


if(selectbox == 'Support Vector Machine'):
    st.sidebar.info('''*Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.*''')
    st.sidebar.info('''*The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.*''')
    st.sidebar.info('''*SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine.*''')

    st.title('SUPPORT VECTOR MACHINE')

    #lsvm_model.fit(X_train, Y_train)

    st.markdown('*Accuracy Score*')
    
    #acc3 = (str(accuracy_score(Y_test, lsvm_model.predict(X_test))*100)+'%')
    st.warning(0.778)

    col1, col2, = st.columns(2)
   
    with col1:
      Pregnancies = st.number_input('Number of Pregnancies')
    with col2:
      Glucose = st.number_input('Glucose Level')
    with col1:
      BP = st.number_input('Blood Pressure value')
    with col2:
      SkinThickness = st.number_input('Skin Thickness value')
    with col1:
      Insulin = st.number_input('Insulin Level')
    with col2:
      BMI = st.number_input('BMI value')
    with col1:
      DPF = st.number_input('Diabetes Pedigree Function value')
    with col2:
      Age = st.number_input('Age of the Person')

    #Code for prediction
    
    Age1=Age+10

    diabetes_diagnosis0 = ''
    diabetes_diagnosis1 = ''

    diab_prediction = lsvm_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
    diab_prediction1 = lsvm_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])

    #Creating a button for prediction
  
    if st.button('Diabetes Test Result'):
      
      if (diab_prediction[0]==0):
        diabetes_diagnosis0 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis0)
             
        if(diab_prediction1[0]==0):          
          diabetes_diagnosis1= 'Not Possible'
             
        else:
          diabetes_diagnosis1='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.subheader('After 10 Years') 
        st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis1)
      else:
        diabetes_diagnosis0='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis0)


if(selectbox == 'Random Forest'):
    st.sidebar.info('''*Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.*''')
    st.sidebar.info('''*Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.*''')
    st.sidebar.info('''*Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.*''')

    st.title('RANDOM FOREST')   

    #rfor_model.fit(X_train, Y_train)

    st.markdown('*Accuracy Score*')
    
    #acc4 = (str(accuracy_score(Y_test, rfor_model.predict(X_test))*100)+'%')
    st.warning( 0.996)

    col1, col2, = st.columns(2)
   
    with col1:
      Pregnancies = st.number_input('Number of Pregnancies')
    with col2:
      Glucose = st.number_input('Glucose Level')
    with col1:
      BP = st.number_input('Blood Pressure value')
    with col2:
      SkinThickness = st.number_input('Skin Thickness value')
    with col1:
      Insulin = st.number_input('Insulin Level')
    with col2:
      BMI = st.number_input('BMI value')
    with col1:
      DPF = st.number_input('Diabetes Pedigree Function value')
    with col2:
      Age = st.number_input('Age of the Person')

    #Code for prediction
    
    Age1=Age+10

    diabetes_diagnosis0 = ''
    diabetes_diagnosis1 = ''

    diab_prediction = rfor_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
    diab_prediction1 = rfor_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])

    #Creating a button for prediction
  
    if st.button('Diabetes Test Result'):
      
      if (diab_prediction[0]==0):
        diabetes_diagnosis0 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis0)
             
        if(diab_prediction1[0]==0):          
          diabetes_diagnosis1= 'Not Possible'
             
        else:
          diabetes_diagnosis1='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.subheader('After 10 Years') 
        st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis1)
      else:
        diabetes_diagnosis0='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis0)


if(selectbox == 'Decision Tree'):
    st.sidebar.info('''*Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.*''')
    st.sidebar.info('''*In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.The decisions or the test are performed on the basis of features of the given dataset.*''')
    st.sidebar.info('''*It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.*''')

    st.title('DECISION TREE')

    #dtree_model.fit(X_train, Y_train)

    st.markdown('*Accuracy Score*')
    
    #acc5 = (str(accuracy_score(Y_test, dtree_model.predict(X_test))*100)+'%')
    st.warning(0.992)

    col1, col2, = st.columns(2)
   
    with col1:
      Pregnancies = st.number_input('Number of Pregnancies')
    with col2:
      Glucose = st.number_input('Glucose Level')
    with col1:
      BP = st.number_input('Blood Pressure value')
    with col2:
      SkinThickness = st.number_input('Skin Thickness value')
    with col1:
      Insulin = st.number_input('Insulin Level')
    with col2:
      BMI = st.number_input('BMI value')
    with col1:
      DPF = st.number_input('Diabetes Pedigree Function value')
    with col2:
      Age = st.number_input('Age of the Person')

    #Code for prediction
    
    Age1=Age+10

    diabetes_diagnosis0 = ''
    diabetes_diagnosis1 = ''

    diab_prediction = dtree_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
    diab_prediction1 = dtree_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])

    #Creating a button for prediction
  
    if st.button('Diabetes Test Result'):
      
      if (diab_prediction[0]==0):
        diabetes_diagnosis0 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis0)
             
        if(diab_prediction1[0]==0):          
          diabetes_diagnosis1= 'Not Possible'
             
        else:
          diabetes_diagnosis1='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.subheader('After 10 Years') 
        st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis1)
      else:
        diabetes_diagnosis0='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis0)


if (selectbox == 'All Algorithms'):

  #st.sidebar.info('''*Algorithms Are Our Used*''')
  st.sidebar.info('''*Logistic Regression*''')
  st.sidebar.info('''*K-Nearest Neighbor*''')
  st.sidebar.info('''*Naive Bayes*''')
  st.sidebar.info('''*Support Vector Machine*''')
  st.sidebar.info('''*Random Forest*''')
  st.sidebar.info('''*Decision Tree*''')

  st.title('ALL ALGORITHMS')

  col1, col2, = st.columns(2)
   
  with col1:
    Pregnancies = st.number_input('Number of Pregnancies')
  with col2:
    Glucose = st.number_input('Glucose Level')
  with col1:
    BP = st.number_input('Blood Pressure value')
  with col2:
    SkinThickness = st.number_input('Skin Thickness value')
  with col1:
    Insulin = st.number_input('Insulin Level')
  with col2:
    BMI = st.number_input('BMI value')
  with col1:
    DPF = st.number_input('Diabetes Pedigree Function value')
  with col2:
    Age = st.number_input('Age of the Person')

  #Code for prediction

  Age1=Age+10

  diabetes_diagnosis0 = ''
  diabetes_diagnosis1 = ''
  diabetes_diagnosis2 = ''
  diabetes_diagnosis3 = ''
  diabetes_diagnosis4 = ''
  diabetes_diagnosis5 = ''
  diabetes_diagnosis6 = ''
  diabetes_diagnosis7 = ''
  diabetes_diagnosis8 = ''
  diabetes_diagnosis9 = ''
  diabetes_diagnosis10 = ''
  diabetes_diagnosis11 = ''


  diab_prediction0 = lr_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
  diab_prediction1 = lr_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])
  diab_prediction2 = knn_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
  diab_prediction3 = knn_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])
  diab_prediction4 = nb_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
  diab_prediction5 = nb_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])
  diab_prediction6 = lsvm_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
  diab_prediction7 = lsvm_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])
  diab_prediction8 = rfor_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
  diab_prediction9 = rfor_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])
  diab_prediction10 = dtree_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]])
  diab_prediction11 = dtree_model.predict([[Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF,Age1]])

  #Creating a button for prediction
  
  if st.button('Diabetes Test Result'):
       
      st.subheader('Logistic Regression (0.780)')

      if (diab_prediction0[0]==0):
        diabetes_diagnosis0 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis0)
             
        if(diab_prediction1[0]==0):          
          diabetes_diagnosis1= 'Not Possible'
             
        else:
          diabetes_diagnosis1='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.markdown('After 10 Years') 
        #st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis1)
      else:
        diabetes_diagnosis0='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis0)

      st.subheader('K-Nearest Neighbor (0.873)')

      if (diab_prediction2[0]==0):
        diabetes_diagnosis2 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis2)
             
        if(diab_prediction3[0]==0):          
          diabetes_diagnosis3= 'Not Possible'
             
        else:
          diabetes_diagnosis3='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.markdown('After 10 Years') 
        #st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis3)
      else:
        diabetes_diagnosis2='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis2)
      
      st.subheader('Naive Bayes (0.763)')

      if (diab_prediction4[0]==0):
        diabetes_diagnosis4 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis4)
             
        if(diab_prediction5[0]==0):          
          diabetes_diagnosis5= 'Not Possible'
             
        else:
          diabetes_diagnosis5='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.markdown('After 10 Years') 
        #st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis5)
      else:
        diabetes_diagnosis4='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis4)

      st.subheader('Support Vector Machine (0.778)')

      if (diab_prediction6[0]==0):
        diabetes_diagnosis6 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis6)
             
        if(diab_prediction7[0]==0):          
          diabetes_diagnosis7= 'Not Possible'
             
        else:
          diabetes_diagnosis7='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.markdown('After 10 Years') 
        #st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis7)
      else:
        diabetes_diagnosis6='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis6)

      st.subheader('Random Forest (0.996)')

      if (diab_prediction8[0]==0):
        diabetes_diagnosis8 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis8)
             
        if(diab_prediction9[0]==0):          
          diabetes_diagnosis9= 'Not Possible'
             
        else:
          diabetes_diagnosis9='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.markdown('After 10 Years') 
        #st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis9)
      else:
        diabetes_diagnosis8='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis8)


      st.subheader('Decision Tree (0.992)')

      if (diab_prediction10[0]==0):
        diabetes_diagnosis10 = 'The person is not Diabetic'
        st.success(diabetes_diagnosis10)
             
        if(diab_prediction11[0]==0):          
          diabetes_diagnosis11= 'Not Possible'
             
        else:
          diabetes_diagnosis11='Possible'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.markdown('After 10 Years') 
        #st.metric(label='Now The Age',value=Age1)
        st.success(diabetes_diagnosis11)
      else:
        diabetes_diagnosis10='The Person is Diabetic'  
        
        st.success(diabetes_diagnosis10)



#completed