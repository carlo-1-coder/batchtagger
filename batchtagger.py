import numpy as np
import pandas as pd
import streamlit as st
import pickle

model = pickle.load(open('final_model.sav', 'rb'))
  
def main(): 
    st.title("Batch Tagger")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Batch Tagger App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
        
    csv_file = st.file_uploader('Upload CSV file', type=['csv'])

    if csv_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Display some statistics of the CSV file
        st.write('**Data Overview:**')
        st.write(df.head())

        st.write('**Data Statistics:**')
        st.write(df.describe())

        prediction = np.array([0])
        if st.button('Make Prediction'):
            prediction = model.predict(df)
            st.write('Prediction:', prediction)

        # Make predictions using the loaded model
        # predictions = model.predict(df)

        # Add the predictions as a new column in the DataFrame
        df['Predictions'] = prediction

        # Offer the modified CSV file for download
        st.write('**Download Prediction CSV File:**')
        modified_csv = df.to_csv(index=False)
        st.download_button(
            label='Download CSV',
            data=modified_csv,
            file_name='prediction.csv',
            mime='text/csv'
        )
      
if __name__=='__main__': 
    main()
