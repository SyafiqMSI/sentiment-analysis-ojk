import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Set page configuration
st.set_page_config(page_title='OJK Survey Dashboard', layout='wide')

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./data/hasil/all_data.csv', sep=';', encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load the data
df = load_data()

# Main dashboard function
def main():
    st.title('OJK Survey Data Dashboard')
    
    # Sidebar for filtering
    st.sidebar.header('Filters')
    
    # Multi-select filters
    if not df.empty:
        # Bidang Filter
        bidang_filter = st.sidebar.multiselect(
            'Select Bidang',
            options=df['BIDANG'].unique().tolist(),
            default=df['BIDANG'].unique().tolist()
        )
        
        # Satker Filter
        satker_filter = st.sidebar.multiselect(
            'Select Satker',
            options=df['SATKER (AKRONIM)'].unique().tolist(),
            default=df['SATKER (AKRONIM)'].unique().tolist()
        )
        
        # Jenis Survei Filter
        jenis_survei_filter = st.sidebar.multiselect(
            'Select Jenis Survei',
            options=df['JENIS SURVEI'].unique().tolist(),
            default=df['JENIS SURVEI'].unique().tolist()
        )
        
        # Label Filter
        label_filter = st.sidebar.multiselect(
            'Select Label',
            options=df['Label'].unique().tolist(),
            default=df['Label'].unique().tolist()
        )
        
        # Apply filters
        filtered_df = df[
            (df['BIDANG'].isin(bidang_filter)) &
            (df['SATKER (AKRONIM)'].isin(satker_filter)) &
            (df['JENIS SURVEI'].isin(jenis_survei_filter)) &
            (df['Label'].isin(label_filter))
        ]
        
        # Metrics Section
        st.header('Survey Metrics')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric('Total Responses', filtered_df.shape[0])
        
        with col2:
            st.metric('Average Confidence', f"{filtered_df['Confidence'].mean():.2f}")
        
        with col3:
            label_counts = filtered_df['Label'].value_counts()
            st.metric('Dominant Label', label_counts.index[0] if len(label_counts) > 0 else 'N/A')
        
        # Visualizations
        st.header('Visualizations')
        
        # Label Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Label Distribution')
            label_fig = px.pie(
                filtered_df, 
                names='Label', 
                title='Distribution of Sentiment Labels'
            )
            st.plotly_chart(label_fig)
        
        with col2:
            st.subheader('Confidence by Label')
            conf_fig = px.box(
                filtered_df, 
                x='Label', 
                y='Confidence', 
                title='Confidence Scores by Label'
            )
            st.plotly_chart(conf_fig)
        
        # Detailed Data View
        st.header('Detailed Data')
        search_term = st.text_input('Search across all columns')
        # st.dataframe(filtered_df)
        if search_term:
            # Convert to lowercase for case-insensitive search
            search_df = filtered_df[
                filtered_df.apply(
                    lambda row: row.astype(str).str.contains(search_term, case=False).any(), 
                    axis=1
                )
            ]
        else:
            search_df = filtered_df
        
        # Display filtered and searched data
        st.dataframe(search_df)
        st.header('Confidence Trend')
        confidence_line = px.line(
            search_df.groupby('Label')['Confidence'].mean().reset_index(), 
            x='Label', 
            y='Confidence', 
            title='Average Confidence by Label',
            markers=True
        )
        st.plotly_chart(confidence_line)
    else:
        st.warning('No data loaded. Please check the data file path.')

# Run the dashboard
if __name__ == '__main__':
    main()