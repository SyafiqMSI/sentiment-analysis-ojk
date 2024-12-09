import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from collections import Counter
import re
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title='OJK Survey Dashboard', layout='wide')

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/hasil/all_data.csv', sep=';', encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def extract_keywords(text, stop_words):
    if pd.isna(text):
        return []
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return words

def get_keyword_options(df, columns, stop_words):
    all_keywords = []
    for column in columns:
        for text in df[column]:
            all_keywords.extend(extract_keywords(text, stop_words))
    
    keyword_counts = Counter(all_keywords)
    
    return sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

def custom_average(row):
    columns = ['RESOURCE PERCEPTION', 'PERFORMANCE DELIVERY', 'OUTCOME SATISFACTION']
    non_zero_count = sum(1 for col in columns if not pd.isna(row[col]) and row[col] != 0)
    if non_zero_count == 0:
        return 0
    
    total = sum(row[col] for col in columns if not pd.isna(row[col]) and row[col] != 0)
    return total / non_zero_count

def fill_missing_with_median(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_value = df[col].median(skipna=True)
        df[col] = df[col].apply(lambda x: median_value if pd.isna(x) or x == 0 else x)
        
def main():
    st.title('OJK Survey Data Dashboard')
    
    try:
        stop_words = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('indonesian'))
    
    df = load_data()
    
    if not df.empty:
        columns_to_fill = ['RESOURCE PERCEPTION', 'PERFORMANCE DELIVERY', 'OUTCOME SATISFACTION']
        fill_missing_with_median(df, columns_to_fill)
        
        st.sidebar.header('Filters')
        
        df['Custom Average'] = df.apply(custom_average, axis=1)
        
        open_questions_keywords = get_keyword_options(
            df, 
            ['OPEN QUESTION 1', 'OPEN QUESTION 2'], 
            stop_words
        )
        
        jenis_survei_filter = st.sidebar.multiselect(
            'Select Jenis Survei',
            options=df['JENIS SURVEI'].unique().tolist(),
            default=df['JENIS SURVEI'].unique().tolist()
        )
        
        # New: Tipe Question Filter
        tipe_question_filter = st.sidebar.multiselect(
            'Select Fungsi',
            options=df['TIPE QUESTION'].unique().tolist(),
            default=df['TIPE QUESTION'].unique().tolist()
        )
        
        bidang_filter = st.sidebar.multiselect(
            'Select Bidang',
            options=df['BIDANG'].unique().tolist(),
            default=df['BIDANG'].unique().tolist()
        )
        
        satker_filter = st.sidebar.multiselect(
            'Select Satker',
            options=df['SATKER (AKRONIM)'].unique().tolist(),
            default=df['SATKER (AKRONIM)'].unique().tolist()
        )
        
        
        label_filter = st.sidebar.multiselect(
            'Select Label',
            options=df['Label'].unique().tolist(),
            default=df['Label'].unique().tolist()
        )
        
        open_questions_keyword_filter = st.sidebar.multiselect(
                'Keywords in Open Questions',
                options=[kw[0] for kw in open_questions_keywords],
                default=[],  
                format_func=lambda x: f"{x} ({dict(open_questions_keywords)[x]} times)"
        )
       
        filtered_df = df[
            (df['TIPE QUESTION'].isin(tipe_question_filter)) &
            (df['BIDANG'].isin(bidang_filter)) &
            (df['SATKER (AKRONIM)'].isin(satker_filter)) &
            (df['JENIS SURVEI'].isin(jenis_survei_filter)) &
            (df['Label'].isin(label_filter))
        ]
        
        if open_questions_keyword_filter:
            filtered_df = filtered_df[
                filtered_df.apply(
                    lambda row: any(
                        kw in extract_keywords(row['OPEN QUESTION 1'], stop_words) or 
                        kw in extract_keywords(row['OPEN QUESTION 2'], stop_words) 
                        for kw in open_questions_keyword_filter
                    ), 
                    axis=1
                )
            ]
        
        st.header('Survey Metrics')
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric('Total Responses', filtered_df.shape[0])
        
        with col2:
            label_counts = filtered_df['Label'].value_counts()
            st.metric('Dominant Label', label_counts.index[0] if len(label_counts) > 0 else 'N/A')
        st.header('Visualizations')
        
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
            st.subheader('Data Count by Label')
            label_count_fig = px.bar(
                filtered_df['Label'].value_counts().reset_index(), 
                x='Label', 
                y='count', 
                title='Number of Data Points by Label'
            )
            st.plotly_chart(label_count_fig)
        
        st.header('Detailed Data')
        search_term = st.text_input('Search across all columns')
        if search_term:
            search_df = filtered_df[
                filtered_df.apply(
                    lambda row: row.astype(str).str.contains(search_term, case=False).any(), 
                    axis=1
                )
            ]
        else:
            search_df = filtered_df
        
        st.dataframe(search_df)
        st.header('Trend Response')
        custom_avg_line = px.line(
            filtered_df.groupby('Label')['Custom Average'].mean().reset_index(), 
            x='Label', 
            y='Custom Average', 
            title='Average score Resource Perception Performance Delivery Outcome Satisfaction',
            markers=True
        )
        st.plotly_chart(custom_avg_line)

        
        if open_questions_keyword_filter:
            st.header('Keyword Analysis')
            
            kw_freq_data = []
            for kw in open_questions_keyword_filter:
                oq1_count = filtered_df['OPEN QUESTION 1'].apply(
                    lambda x: kw in extract_keywords(x, stop_words)
                ).sum()
                oq2_count = filtered_df['OPEN QUESTION 2'].apply(
                    lambda x: kw in extract_keywords(x, stop_words)
                ).sum()
                
                kw_freq_data.append({
                    'Keyword': kw,
                    'Open Question 1 Count': oq1_count,
                    'Open Question 2 Count': oq2_count
                })
            
            kw_freq_df = pd.DataFrame(kw_freq_data)
            
            st.dataframe(kw_freq_df, use_container_width=True)
            
            fig = go.Figure(data=[
                go.Bar(name='Open Question 1', x=kw_freq_df['Keyword'], y=kw_freq_df['Open Question 1 Count']),
                go.Bar(name='Open Question 2', x=kw_freq_df['Keyword'], y=kw_freq_df['Open Question 2 Count'])
            ])
            fig.update_layout(barmode='group', title='Keyword Frequencies in Open Questions')
            st.plotly_chart(fig)

        
    
    else:
        st.warning('No data loaded. Please check the data file path.')

if __name__ == '__main__':
    main()