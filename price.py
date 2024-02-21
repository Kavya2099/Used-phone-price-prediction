import pickle
import streamlit as st
import pandas as pd
import numpy as np

#from streamlit_option_menu import option_menu

# loading the saved model

price_model = pickle.load(open('finalized_model.sav', 'rb'))


# page title
st.title('Used Phone Price Prediction using ML')

st.markdown(
    f'<div style="display: flex; justify-content: center;"><img src="https://i.pinimg.com/originals/70/7c/39/707c39bfff546612b5b4604fe86cda32.gif" width="300"></div>', 
    unsafe_allow_html=True,
)


# Define the categories for each column
device_categories = ('Others', 'Samsung', 'Huawei', 'LG', 'Lenovo', 'ZTE', 'Xiaomi', 'Oppo', 'Asus', 'Alcatel',
                     'Micromax', 'Vivo', 'Honor', 'HTC', 'Nokia', 'Motorola', 'Sony', 'Meizu', 'Gionee', 'Acer',
                     'XOLO', 'Panasonic', 'Realme', 'Apple', 'Lava', 'Celkon', 'Spice', 'Karbonn', 'Coolpad',
                     'BlackBerry', 'Microsoft', 'OnePlus', 'Google', 'Infinix')
os_categories = ('Android', 'Others', 'iOS', 'Windows')
_4g_categories = ('yes', 'no')
_5g_categories = ('yes', 'no')

# Display select boxes for each categorical column
device_type = st.selectbox('Device type', device_categories)
os_type = st.selectbox('OS type', os_categories)
screen_size = st.number_input('Screen size in cm')
_4g = st.selectbox('4G supported', _4g_categories)
_5g = st.selectbox('5G supported', _5g_categories)
rear_camera_mp = st.number_input('Rear camera value in megapixels')
front_camera_mp = st.number_input('Front camera value in megapixels')
internal_memory = st.number_input('Internal memory(ROM) in GB',step=1)
ram = st.number_input('RAM in GB',step=1)
battery = st.number_input('Energy capacity of the device battery in mAh',step=1)
weight = st.number_input('Weight of the device in grams',step=1)
release_year = st.number_input("Year when the device model was released",step=1)
days_used= st.number_input("Number of days the used/refurbished device has been used",step=1)
new_price=st.number_input("Enter the price new device of the same model",step=1)
features_values = {'device_type': device_type, 'os_type': os_type, '4g': _4g, '5g': _5g, 
                   'screen_size': screen_size, 'Rear_camera_mp': rear_camera_mp, 
                   'Front_camera_mp': front_camera_mp, 'Internal_memory': internal_memory, 
                   'RAM': ram, 'Battery': battery, 'Weight': weight, 
                   'Release_year': release_year, 'Days_used': days_used, 
                   'new_price': new_price}

if st.button('Submit'):
    if any(value == 0 or value == 0.00 for value in features_values.values()):
        st.warning('Please input all the details.')
    else:
        normalized_new_price=np.log(new_price)
    # Create a DataFrame with the input values
        data = pd.DataFrame({
            'Device_type': [device_type],
            'OS_type': [os_type],
            '4g': [_4g],
            '5g': [_5g],
            'screen_size': [screen_size],
            'Rear_camera_mp': [rear_camera_mp],
            'Front_camera_mp': [front_camera_mp],
            'Internal_memory': [internal_memory],
            'RAM': [ram],
            'Battery': [battery],
            'Weight': [weight],
            'Release_year': [release_year],
            'Days_used': [days_used],
            'Normalized_new_price': [normalized_new_price]
        })
        data_1 = pd.DataFrame({'screen_size': [screen_size],
            'Front_camera_mp': [front_camera_mp],
            'Internal_memory': [internal_memory],
            'Battery': [battery],
            'Normalized_new_price': [normalized_new_price]
        })
        
        # Perform one-hot encoding for categorical variables
        data_encoded = pd.get_dummies(data_1)
        
        # Predict the used price using the model
        prediction = price_model.predict(data_encoded)

        if new_price<=10000:
            prediction = np.exp(prediction)*7
        elif new_price>10000 and new_price<=16000:
            prediction = np.exp(prediction)*10
        elif new_price>16000 and new_price<=22000:
            prediction = np.exp(prediction)*13
        elif new_price>22000 and new_price<=27000:
            prediction = np.exp(prediction)*16
        elif new_price>27000 and new_price<=49999:
            prediction = np.exp(prediction)*25
        elif new_price>=50000 and new_price<=70000:
            prediction = np.exp(prediction)*30
        elif new_price>70000 and new_price<=1000000:
            prediction = np.exp(prediction)*40

        st.success(f'Price of the used/refurbished device: Rs.{round(prediction[0], 2)}')
