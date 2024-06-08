import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go
import os

# Function to get the absolute path of the dataset
def get_dataset_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, '..', 'data', 'Breast Cancer Diagnosis Dataset.csv')

# Load the dataset using the absolute path
dataset_path = get_dataset_path()

if os.path.exists(dataset_path):
    data = pd.read_csv(dataset_path)
else:
    st.error(f"File not found: {dataset_path}")
    st.stop()

# Encode the target variable
le = LabelEncoder()
data['Class'] = le.fit_transform(data['Class'])

# Split the data into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=50)
model.fit(X_train, y_train)

# Streamlit app
st.title('Breast Cancer Diagnosis Dashboard')

# Display model performance
st.subheader('Model Performance')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')

# Custom labels for the input features
custom_labels = {
    'Start Age': 'Start Age:',
    'End Age': 'End Age:',
    'menopause': 'Menopause Status (1 = It40 | 2 = ge40 | 3 = premeno):',
    'Start tumor size': 'Tumor Size From:',
    'End tumor size': 'Tumor Size To:',
    'Start_env_nodes': 'Involved Nodes From:',
    'end_env_nodes': 'Involved Nodes To:',
    'node-caps': 'Node Capsular Status (1 = Yes | 0 = No): ',
    'deg-malig': 'Degree of Malignancy:',
    'breast': 'Affected Breast Side (1 = Right | 2 = Left):',
    'breast-quad': 'Affected Breast Quadrant (1 = Central | 2 = Left Up | 3 = Left Low | 4 = Right Up | 5 = Right Low):',
    'irradiat': 'Received Radiation Treatment (1 = Yes | 0 = No):'
}

# User input features
st.sidebar.subheader('Input Patient Features')
features = X.columns.tolist()
input_data = {}
for feature in features:
    label = custom_labels.get(feature, feature)  # Use custom label if available
    # Check if feature is categorical or numerical
    if X[feature].dtype == 'object':
        input_data[feature] = st.sidebar.selectbox(f'Select {label}', X[feature].unique())
    else:
        input_value = st.sidebar.number_input(f'Enter {label}', value=int(X[feature].mean()), step=1)
        # Custom validation for specific features
        if feature == 'irradiat' and input_value not in [0, 1]:
            st.sidebar.error('Please enter either 0 or 1 for Radiation Treatment Received.')
            st.stop()
        elif feature == 'breast-quad' and input_value not in [1, 2, 3, 4, 5]:
            st.sidebar.error('Please enter a number between 1 and 5 for Affected Breast Quadrant')
            st.stop()
        elif feature == 'breast' and input_value not in [1, 2]:
            st.sidebar.error('Please enter either 1 or 2 for Affected Breast Side.')
            st.stop()
        elif feature == 'deg-malig' and input_value not in [1, 2, 3]:
            st.sidebar.error('Please enter a number between 1 and 3 for Degree of Malignancy.')
            st.stop()
        elif feature == 'node-caps' and input_value not in [0, 1]:
            st.sidebar.error('Please enter either 0 or 1 for Node Capsular Status.')
            st.stop()
        elif feature == 'menopause' and input_value not in [1, 2, 3]:
            st.sidebar.error('Please enter a number between 1 and 3 for Menopause.')
            st.stop()
        else:
            input_data[feature] = input_value

# Make prediction
if st.sidebar.button('Predict'):
    input_values = [input_data[feature] for feature in features]
    prediction_proba = model.predict_proba([input_values])
    no_recurrence_prob = prediction_proba[0][0]
    recurrence_prob = prediction_proba[0][1]
    
    if recurrence_prob > no_recurrence_prob:
        result = f'The patient is likely to have <br><div style="background-color:#ff3b89; padding: 10px; border-radius: 5px; color: #000000;"><b> Recurrence Events</b></div><br>'
        result += f'<div style="background-color:; padding: 10px; border-radius: 5px; color: #000000;">The probability of <b>Recurrence Event</b> is {recurrence_prob:.5f}.<br> The probability of <b>No Recurrence Event</b> is {no_recurrence_prob:.5f}.</div>'
    else:
        result = f'The patient is likely to have <div style="background-color:#0bdb81; padding: 10px; border-radius: 5px; color: #000000;"><b>No Recurrence Events</b></div><br>'
        result += f'<div style="background-color:; padding: 10px; border-radius: 5px; color: #000000;">The probability of <b>No Recurrence Event</b> is {no_recurrence_prob:.5f}.<br> The probability of <b>Recurrence Event</b> is {recurrence_prob:.5f}.</div>'
    
    st.markdown(result, unsafe_allow_html=True)

# Function to generate radar chart
def generate_radar_chart(input_data):
    # Define radar chart labels
    labels = list(input_data.keys())
    
    # Define radar chart values
    values = list(input_data.values())
    
    # Create radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Patient Data'
    ))
    
    # Add title and layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values)]
            )
        ),
        title="",
    )
    
    return fig

# Display radar chart
st.subheader('Radar Chart')
radar_chart = generate_radar_chart(input_data)
st.plotly_chart(radar_chart)

# Create confusion matrix
st.subheader('Confusion Matrix')
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with smaller size
fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figure size here
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['No Recurrence', 'Recurrence'])
ax.yaxis.set_ticklabels(['No Recurrence', 'Recurrence'])

# Display the plot in Streamlit
st.pyplot(fig)

# Add background color to specific elements
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"]{
        background-color: #FFCECE;
        opacity: 0.9;
        background-image: radial-gradient(#e02761 0.5px, #FFCECE 0.5px);
        background-size: 10px 10px;
    }

    /* Title */
    .stMarkdown h1 
    {
        color: #902E4D;
        text-shadow: 2px 2px 2px #902E4D;
    }

    /* Subheaders */
        .stMarkdown h2 {
        color: #000000;
        text-shadow: 1px 1px 1px #000000;
    }

    /* Subheaders */
        .stMarkdown h3 {
        color: #000000;
        text-shadow: 1px 1px 1px #000000;
    }

    /* Subheaders */
        .stMarkdown p {
        color: #000000;
        text-shadow: 1px 1px 1px #000000;
    }

    .st-b8 {
        background-color: #F45E92 !important;
    }

    .st-br {
        background-color: #F45E92 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #C8496F;
    }

    /* Input fields */
    .stNumberInput input {
        background-color: #DD8CA4 !important;
        color: #ffffff;
    }

    .stSelectbox div {
        background-color: #FFF0F5 !important;
        color: #FF9EC6;
    }

    .stButton > button {
        background-color: #FFA07A !important;
        color: #000000;
        display: block;
        margin: 0 auto;
        text-align: center;
    }

    .stButton > button:hover {
        background-color: #FF6347 !important;
        color: #F8BCB2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)