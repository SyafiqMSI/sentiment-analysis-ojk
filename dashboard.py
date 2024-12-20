import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from collections import Counter
import re
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title='Survey Dashboard', layout='wide', page_icon="📊")

@st.cache_data
def load_data(file_path, separator=';', encoding='utf-8'):
    try:
        df = pd.read_csv(file_path, sep=separator, encoding=encoding)
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

def get_keyword_frequency(df, column, stop_words):
    all_keywords = []
    for text in df[column]:
        all_keywords.extend(extract_keywords(text, stop_words))
    
    keyword_counts = Counter(all_keywords)
    
    keyword_df = pd.DataFrame(
        keyword_counts.items(), 
        columns=['Keyword', 'Frequency']
    ).sort_values('Frequency', ascending=False)

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
    for text in df['Text']:
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

def calculate_sentiment_weight(df, filter_columns=None):
    if df.empty:
        return 0
    
    if filter_columns and isinstance(filter_columns, dict):
        filtered_df = df.copy()
        for column, selected_options in filter_columns.items():
            if selected_options and len(selected_options) > 0:
                filtered_df = filtered_df[filtered_df[column].isin(selected_options)]
    else:
        filtered_df = df
    
    label_map = {
        1: "sangat tidak setuju",
        2: "tidak setuju",
        3: "kurang setuju",
        4: "cukup setuju",
        5: "setuju",
        6: "sangat setuju"
    }
    
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    filtered_df['Label_Index'] = filtered_df['Label'].map(reverse_label_map)
    
    def calculate_weight(index):
        if index == 0:
            return 1
        elif 0.1 <= index <= 3.9:
            return index + 0.73 
        elif 4 <= index <= 6:
            return 6
        else:
            return None
    
    if 'Adjustment' in df['Title'].iloc[0]:
        filtered_df['NILAI_SENTIMEN'] = filtered_df['Label_Index'].apply(calculate_weight)
    elif 'IDI Survey' in df['Title'].iloc[0]:
        filtered_df['NILAI_SENTIMEN'] = filtered_df['Label_Index'].apply(calculate_weight)
    else:
        filtered_df['NILAI_SENTIMEN'] = filtered_df['Label_Index']
    
    average_bobot_sentimen = filtered_df['NILAI_SENTIMEN'].mean()
    
    return average_bobot_sentimen

def create_survey_dashboard(df, title, stop_words, open_question_columns):
    
    df['Title'] = title
    
    st.title(title)
    
    if not df.empty:
        st.sidebar.header('Cascading Filters')
        
        filter_columns = [
            col for col in ['JENIS SURVEI', 'TIPE QUESTION', 'BIDANG', 'SATKER (AKRONIM)', 'Label'] 
            if col in df.columns
        ]
        
        selected_filters = {}
        
        filtered_df = df.copy()
        
        for i, column in enumerate(filter_columns):
            available_options = filtered_df[column].unique().tolist()
            
            filter_key = f'filter_{column}_{title}'
            selected_options = st.sidebar.multiselect(
                f'Select {column}',
                options=available_options,
                default=available_options,
                key=filter_key
            )
            
            selected_filters[column] = selected_options
            
            filtered_df = filtered_df[filtered_df[column].isin(selected_options)]
        
        keywords = get_keyword_options(
            filtered_df, 
            open_question_columns, 
            stop_words
        )
        
        keyword_filter = st.sidebar.multiselect(
            'Keywords in Open Questions',
            options=[kw[0] for kw in keywords],
            default=[],  
            format_func=lambda x: f"{x} ({dict(keywords)[x]} times)"
        )
        
        if keyword_filter:
            filtered_df = filtered_df[
                filtered_df.apply(
                    lambda row: any(
                        kw in extract_keywords(row[col], stop_words) 
                        for col in open_question_columns 
                        for kw in keyword_filter
                    ), 
                    axis=1
                )
            ]
            
        filter_columns_dict = {
            col: selected_filters.get(col, []) for col in filter_columns
        }

        average_sentiment_weight = calculate_sentiment_weight(
            df, 
            filter_columns_dict  
        )
        
        st.header('Survey Metrics')
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric('Total Responses', filtered_df.shape[0])

        with col2:
            label_counts = filtered_df['Label'].value_counts()
            st.metric('Dominant Label', label_counts.index[0] if len(label_counts) > 0 else 'N/A')
            
        with col3:
            if filtered_df.empty:
                average_sentiment_weight = 0  
            if title == "Adjustment Factor 2 Open Question":
                st.metric('Nilai Bobot Sentimen', f"{round(average_sentiment_weight):.2f}")
            elif title == "Adjustment Factor 1 Open Question":
                st.metric('Nilai Bobot Sentimen', f"{round(average_sentiment_weight):.2f}")
            elif title == "Gabungan":
                if 'AF_AVERAGE' in filtered_df.columns:
                    af_average = filtered_df['AF_AVERAGE'].mean()
                    st.metric('Nilai Rata-rata AF', f"{af_average:.2f}")
                else:
                    st.metric('Nilai Rata-rata AF', 'N/A')
            else:
                st.metric('Nilai Bobot Sentimen', f"{average_sentiment_weight:.2f}")

        
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
        search_term = st.text_input('Search Tabel Query', placeholder="Search across all columns", key=f"search_input_{title}")

        if search_term:
            search_df = filtered_df[
                filtered_df.apply(
                    lambda row: row.astype(str).str.contains(search_term, case=False).any(),
                    axis=1
                )
            ]
        else:
            search_df = filtered_df

        # columns_to_exclude = ['New_Label', 'Confidence', 'NAMA PIC/RESPONDEN', 'EMAIL', 'KONTAK', 'EMAIL CADANGAN', 'KOTAK CADANGAN', 'Combined_Text']
        # display_df = search_df.drop(columns=columns_to_exclude, errors='ignore')
        
        columns_to_exclude = ['New_Label', 'Confidence', 'NAMA PIC/RESPONDEN', 'EMAIL', 'KONTAK', 'EMAIL CADANGAN', 'KOTAK CADANGAN', 'Combined_Text', 'Text', 'Title']

        if title == "Confirmation Factor 2 Open Question":
            columns_to_exclude.append('NILAI_SENTIMEN')
            
        if title == "Confirmation Factor 1 Open Question":
            columns_to_exclude.append('NILAI_SENTIMEN')

        display_df = search_df.drop(columns=columns_to_exclude, errors='ignore')
        st.dataframe(display_df)
        
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
        
        if keyword_filter:
            st.header('Keyword Analysis')
            
            kw_freq_data = []
            for kw in keyword_filter:
                kw_freq_row = {'Keyword': kw}
                for col in open_question_columns:
                    count = filtered_df[col].apply(
                        lambda x: kw in extract_keywords(x, stop_words)
                    ).sum()
                    kw_freq_row[f'{col} Count'] = count
                
                kw_freq_data.append(kw_freq_row)
            
            kw_freq_df = pd.DataFrame(kw_freq_data)
            st.dataframe(kw_freq_df, use_container_width=True)
            
            bar_data = []
            for col in open_question_columns:
                bar_data.append(
                    go.Bar(
                        name=col, 
                        x=kw_freq_df['Keyword'], 
                        y=kw_freq_df[f'{col} Count']
                    )
                )
            
            fig = go.Figure(data=bar_data)
            fig.update_layout(barmode='group', title='Keyword Frequencies in Open Questions')
            st.plotly_chart(fig)
    
    else:
        st.warning('No data loaded. Please check the data file path.')

        
def idi_page():
    st.title('IDI Survey Data Dashboard')
    idi_df = load_data('data/hasil/main_data_idi_sentimen.csv')
    
    if not idi_df.empty:
     
        try:
            stop_words = set(stopwords.words('indonesian'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('indonesian'))
        
        st.sidebar.header('Cascading Filters')
        
        idi_df['Title'] = 'IDI Survey Dashboard'
        
        jenis_filter = st.sidebar.multiselect(
            'Select Jenis',
            options=idi_df['Jenis'].unique().tolist(),
            default=idi_df['Jenis'].unique().tolist(),
            key='jenis_filter'
        )
        
        filtered_df_jenis = idi_df[idi_df['Jenis'].isin(jenis_filter)]
        
        available_labels = filtered_df_jenis['Label'].unique().tolist()
        label_filter = st.sidebar.multiselect(
            'Select Label',
            options=available_labels,
            default=available_labels,
            key='label_filter_idi'
        )
        
        filtered_df = filtered_df_jenis[filtered_df_jenis['Label'].isin(label_filter)]

        filter_columns = {
            'Jenis': jenis_filter,
            'Label': label_filter
        }
        
        idi_keywords = get_keyword_options(
            filtered_df, 
            ['Kualitas Layanan', 'Hubungan dengan OJK', 'Kualitas SDM', 'Saran'], 
            stop_words
        )
        
        idi_keyword_filter = st.sidebar.multiselect(
            'Keywords in IDI',
            options=[kw[0] for kw in idi_keywords],
            default=[],  
            format_func=lambda x: f"{x} ({dict(idi_keywords)[x]} times)"
        )
        
        if idi_keyword_filter:
            filtered_df = filtered_df[
                filtered_df.apply(
                    lambda row: any(
                        kw in extract_keywords(row['Kualitas Layanan'], stop_words) or 
                        kw in extract_keywords(row['Hubungan dengan OJK'], stop_words) or
                        kw in extract_keywords(row['Kualitas SDM'], stop_words) or
                        kw in extract_keywords(row['Saran'], stop_words) 
                        for kw in idi_keyword_filter
                    ), 
                    axis=1
                )
            ]
        
        average_sentiment_weight_idi = calculate_sentiment_weight(
            idi_df, 
            filter_columns
        )
        
        st.header('Survey Metrics')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric('Total Responses', filtered_df.shape[0])
        
        with col2:
            label_counts = filtered_df['Label'].value_counts()
            st.metric('Dominant Label', label_counts.index[0] if len(label_counts) > 0 else 'N/A')
         
        with col3:
            if filtered_df.empty:
                average_sentiment_weight = 0  
            else: 
                st.metric('Nilai Bobot Sentimen', f"{round(average_sentiment_weight_idi):.2f}")

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
        search_placeholder = "Search across all columns"

        search_term = st.text_input('Search Tabel Query', placeholder=search_placeholder, key="search_input_idi")

        if search_term:
            search_df = filtered_df[
                filtered_df.apply(
                    lambda row: row.astype(str).str.contains(search_term, case=False).any(),
                    axis=1
                )
            ]
        else:
            search_df = filtered_df

        columns_to_exclude = ['New_Label','Confidence','Title']  
        display_df = search_df.drop(columns=columns_to_exclude, errors='ignore')  
        st.dataframe(display_df)

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

        if idi_keyword_filter:
            st.header('Keyword Analysis')
            
            kw_freq_data = []
            for kw in idi_keyword_filter:
                oq1_count = filtered_df['Kualitas Layanan'].apply(
                    lambda x: kw in extract_keywords(x, stop_words)
                ).sum()
                oq2_count = filtered_df['Hubungan dengan OJK'].apply(
                    lambda x: kw in extract_keywords(x, stop_words)
                ).sum()
                oq3_count = filtered_df['Kualitas SDM'].apply(
                    lambda x: kw in extract_keywords(x, stop_words)
                ).sum()
                oq4_count = filtered_df['Saran'].apply(
                    lambda x: kw in extract_keywords(x, stop_words)
                ).sum()
                
                kw_freq_data.append({
                    'Keyword': kw,
                    'Kualitas Layanan Count': oq1_count,
                    'Hubungan dengan OJK Count': oq2_count,
                    'Kualitas SDM Count': oq3_count,
                    'Saran Count': oq4_count
                })
            
            kw_freq_df = pd.DataFrame(kw_freq_data)
            
            st.dataframe(kw_freq_df, use_container_width=True)
            
            fig = go.Figure(data=[
                go.Bar(name='Kualitas Layanan', x=kw_freq_df['Keyword'], y=kw_freq_df['Kualitas Layanan Count']),
                go.Bar(name='Hubungan dengan OJK', x=kw_freq_df['Keyword'], y=kw_freq_df['Hubungan dengan OJK Count']),
                go.Bar(name='Kualitas SDM', x=kw_freq_df['Keyword'], y=kw_freq_df['Kualitas SDM Count']),
                go.Bar(name='Saran', x=kw_freq_df['Keyword'], y=kw_freq_df['Saran Count'])
            ])
            fig.update_layout(barmode='group', title='Keyword Frequencies')
            st.plotly_chart(fig)
    
    else:
        st.warning('No IDI data loaded. Please check the data file path.')

pages = {
    "INTERNAL-EKSTERNAL": create_survey_dashboard,
    "IDI Dashboard": idi_page
}

def main():
    try:
        stop_words = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('indonesian'))
    
    st.sidebar.title("Navigation")
    
    datasets = {
        "Gabungan": {
            'path': 'data/hasil/hasil_gabungan.csv',
            'open_questions': ['OPEN QUESTION 1', 'OPEN QUESTION 2']
        },
        "Adjustment Factor 2 Open Question": {
            'path': 'data/hasil/main_data_19_OQ2_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1', 'OPEN QUESTION 2']
        },
        "Adjustment Factor 1 Open Question": {
            'path': 'data/hasil/main_data_19_OQ1_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1']
        },
        "Confirmation Factor 2 Open Question": {
            'path': 'data/hasil/main_data_19_OQ2_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1', 'OPEN QUESTION 2']
        },
        "Confirmation Factor 1 Open Question": {
            'path': 'data/hasil/main_data_19_OQ1_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1']
        }
    }
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a Page", list(pages.keys()))
    
    if page == "IDI Dashboard":
        pages[page]()
    else:
        data = st.sidebar.selectbox("Select a Dataset", list(datasets.keys()))
        dataset_info = datasets[data]
        df = load_data(dataset_info['path'])
        
        pages[page](
            df, 
            data, 
            stop_words, 
            dataset_info['open_questions']
        )

if __name__ == '__main__':
    main()