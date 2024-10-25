# Personalised Marketing Recommendations with OpenAI and Streamlit

This application provides personalised marketing recommendations based on customer data using K-means clustering and the OpenAI API. The application is built with **Streamlit**, allowing users to visualise customer segments and receive targeted marketing strategies. This README covers setup, usage, and feature highlights.

## Features

- **Customer Data Clustering**: Uses K-means to segment customers based on selected attributes.
- **Customised Marketing Recommendations**: Generates marketing strategies tailored to each customer segment through OpenAI.
- **Interactive Visualisation**: Displays data and profile distributions for each customer cluster with Plotly charts.
- **Selective API Usage**: Recommendations are generated only when requested, saving on API costs.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/marketing-recommendations.git
    cd marketing-recommendations
    ```

2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key in the app when prompted in the sidebar.

## Usage

1. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Upload your customer data file (CSV format) in the sidebar and choose the appropriate delimiter for your file.

3. Select the attributes for analysis and clustering.

4. Generate customer segments by clicking on **Identify Customer Profiles**.

5. View the distribution of each selected attribute across customer segments.

6. Click on **Generate Marketing Recommendations** to receive targeted marketing advice for each customer profile.

## Configuration

- **CSV File Upload**: The app allows you to upload CSV files with custom delimiters for flexible data import.
- **OpenAI API Key**: Enter your OpenAI API key in the sidebar to enable recommendations.

## Dependencies

- **Streamlit**
- **Pandas**
- **Plotly**
- **Scikit-learn**
- **OpenAI API**

Install all dependencies using:
```bash
pip install -r requirements.txt
