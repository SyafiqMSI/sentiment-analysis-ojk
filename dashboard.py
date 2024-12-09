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

def extract_action_phrases(text):
    if pd.isna(text):
        return []
    
    text = re.sub(r'\s+', ' ', str(text).lower())
    text = re.sub(r'[^\w\s]', '', text)
    
    conjunctions = ['dan', 'atau', 'serta', 'dengan', 'tetapi', 'namun', 'melainkan', 'sebaliknya','yang']
    
    matches = []
    
    words = text.split()
    
    for i in range(len(words)-2):
        if words[i+1] in conjunctions:
            full_phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            matches.append(full_phrase)
    
    additional_patterns = [
        r'pe\w+an\s\w+',     
        r'me\w+kan\s\w+',    
        r'di\w+kan\s\w+',    
    ]
    
    for pattern in additional_patterns:
        matches.extend(re.findall(pattern, text))
    
    return matches

def analyze_action_phrases(df):
    all_phrases = []
    for text in df['Combined_Text']:
        all_phrases.extend(extract_action_phrases(text))
    
    phrase_counts = Counter()
    for phrase in all_phrases:
        normalized_phrase = re.sub(r'\s+', ' ', phrase).strip()
        phrase_counts[normalized_phrase] += 1
    
    total_phrases = sum(phrase_counts.values())
    
    phrase_data = []
    for phrase, count in phrase_counts.items():
        percentage = (count / total_phrases) * 100
        phrase_data.append({
            'keyword': phrase,
            'jumlah': count,
            'presentase': f"{percentage:.2f}%"
        })
    
    phrase_data.sort(key=lambda x: x['jumlah'], reverse=True)
    
    return pd.DataFrame(phrase_data)

def calculate_sentiment_weight(df):
    label_map = {
        1: "sangat tidak setuju",
        2: "tidak setuju",
        3: "kurang setuju",
        4: "cukup setuju",
        5: "setuju",
        6: "sangat setuju"
    }
    
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    df['Label_Index'] = df['Label'].map(reverse_label_map)
    
    def calculate_weight(index):
        if index == 0:
            return 1
        elif 0.1 <= index <= 3.9:
            return index + 0.73 
        elif 4 <= index <= 6:
            return 6
        else:
            return None
    
    df['NILAI_BOBOT'] = df['Label_Index'].apply(calculate_weight)
    
    average_bobot_sentimen = df['NILAI_BOBOT'].mean()
    
    return average_bobot_sentimen

def main():
    st.title('OJK Survey Data Dashboard')
    
    try:
        stop_words = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('indonesian'))
    
    df = load_data()
    
    if not df.empty:
        
        st.sidebar.header('Filters')
        
        open_questions_keywords = get_keyword_options(
            df, 
            ['OPEN QUESTION 1', 'OPEN QUESTION 2'], 
            stop_words
        )
        
        jenis_survei_filter = st.sidebar.multiselect(
            'Select Jenis Survei',
            options=df['JENIS SURVEI'].unique().tolist(),
            default=df['JENIS SURVEI'].unique().tolist(),
            key='jenis_survei_filter'
        )
        
        available_tipe_questions = df[
            df['JENIS SURVEI'].isin(jenis_survei_filter)
        ]['TIPE QUESTION'].unique().tolist()
        
        tipe_question_filter = st.sidebar.multiselect(
            'Select Fungsi',
            options=available_tipe_questions,
            default=available_tipe_questions,
            key='tipe_question_filter'
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
            
        average_sentiment_weight = calculate_sentiment_weight(df)
        
        st.header('Survey Metrics')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric('Total Responses', filtered_df.shape[0])
        
        with col2:
            label_counts = filtered_df['Label'].value_counts()
            st.metric('Dominant Label', label_counts.index[0] if len(label_counts) > 0 else 'N/A')
         
        with col3:
            st.metric('Avg Nilai Bobot Sentimen', f"{average_sentiment_weight:.2f}")

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

        st.header('Word Analysis')
        verb_active_df = analyze_action_phrases(search_df)
        
        if not verb_active_df.empty:
            st.dataframe(verb_active_df, use_container_width=True)
            
            st.subheader('Top Words')
            top_words_fig = px.bar(
                verb_active_df.head(10), 
                x='keyword', 
                y='jumlah', 
                title='Top 10 Words'
            )
            st.plotly_chart(top_words_fig)
        else:
            st.write("No verb or active words found in the filtered dataset.")

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