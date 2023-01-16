# importat librerias
import streamlit as st 
import pickle
import pandas as pd 


# data extraction

with open('Lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)


with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)


with open('svc_model.pkl', 'rb') as sv:
    svc_model = pickle.load(sv)


#Clasificar las plantas
def classsify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolor'
    else:
        return 'Virginica'


def main():
    st.title('Modelamiento de Iris')

    st.sidebar.header('User Input Parameters')

    def user_input_parameters():
        
        sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

        data ={
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        features = pd.DataFrame(data, index=[0])

        return features

    df = user_input_parameters()

    #choose the model
    option = ['Linear Regression', 'Logistic Regression', 'SVC']
    model = st.sidebar.selectbox('Wich model you like to use? ', option)


    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    #Make the predict
    if st.button('RUN'):
        if model == 'Linear Regression': 
            st.success(classsify(lin_reg.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classsify(log_reg.predict(df)))
        else:
            st.success(classsify(svc_model.predict(df)))
    

if __name__ == '__main__':
    main()