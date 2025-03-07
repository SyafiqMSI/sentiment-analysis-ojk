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

def calculate_sentiment_weight(df, filter_columns=None, is_idi_cf=False):
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
    elif is_idi_cf:
        filtered_df['NILAI_SENTIMEN'] = filtered_df['Label_Index']
    elif 'IDI Survey' in df['Title'].iloc[0]:
        filtered_df['NILAI_SENTIMEN'] = filtered_df['Label_Index'].apply(calculate_weight)
    else:
        filtered_df['NILAI_SENTIMEN'] = filtered_df['Label_Index']

    average_bobot_sentimen = filtered_df['NILAI_SENTIMEN'].mean()
    
    return average_bobot_sentimen

Title = "ojk1103"

def display_topic_analysis(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2.csv', sep=';')
        
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")
        
def display_topic_analysis_ARK(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis ARK')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2_ARK.csv', sep=';')
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")
        
def display_topic_analysis_IAKD(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis IAKD')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2_IAKD.csv', sep=';')
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")
        
def display_topic_analysis_KS(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis KS')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2_KS.csv', sep=';')
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")
        
def display_topic_analysis_MS(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis MS')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2_MS.csv', sep=';')
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")
        
def display_topic_analysis_PBKN(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis PBKN')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2_PBKN.csv', sep=';')
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")

def display_topic_analysis_PEPK(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis PEPK')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2_PEPK.csv', sep=';')
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")

def display_topic_analysis_PMDK(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis PMDK')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2_PMDK.csv', sep=';')
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")

def display_topic_analysis_PPDP(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis PPDP')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2_PPDP.csv', sep=';')
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")

def display_topic_analysis_PVML(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis PVML')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_OQ2_PVML.csv', sep=';')
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Men  yesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")

def display_topic_analysis_kojk(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_KOJKOQ2.csv', sep=';')
        
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Menyesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")
        
def display_topic_analysis_idi(df):
    """
    Menampilkan analisis topik dari pasangan kata kerja-kata benda
    """
    st.header('Topic Analysis')
    
    # Membaca file CSV hasil analisis pasangan kata kerja-kata benda
    try:
        pairs_df = pd.read_csv('./data/hasil/topic_analysis_idi_OQ2.csv', sep=';')
        
        
        
        # Memformat persentase
        pairs_df['Percentage'] = pairs_df['Percentage'].round(2).astype(str) + '%'
        
        # Mengatur urutan kolom dan nama kolom yang ditampilkan
        display_df = pairs_df.rename(columns={
            'Topic': 'Topik',
            'Frequency': 'Frekuensi',
            'Percentage': 'Persentase',
        })
        
        # Menampilkan tabel
        st.dataframe(
            display_df,
            column_config={
                "Topik": st.column_config.TextColumn(
                    "Topik",
                    width="medium"
                ),
                "Frekuensi": st.column_config.NumberColumn(
                    "Frekuensi",
                    format="%d"
                ),
                "Persentase": st.column_config.TextColumn(
                    "Persentase",
                    width="small"
                )
            },
            use_container_width=True
        )
        
        # Menampilkan visualisasi distribusi pasangan kata kerja-kata benda
        pairs_counts = pairs_df.groupby('Topic')['Frequency'].sum().reset_index()
        pairs_counts = pairs_counts.sort_values('Frequency', ascending=True)  # Sort ascending untuk tampilan yang lebih baik
        
        # Hitung persentase untuk setiap pasangan
        total_freq = pairs_counts['Frequency'].sum()
        pairs_counts['Percentage'] = (pairs_counts['Frequency'] / total_freq * 100).round(2)
        
        # Buat bar chart horizontal
        fig = px.bar(
            pairs_counts,
            y='Topic',
            x='Frequency',
            orientation='h',
            title='Distribution of Verb-Noun Pairs',
            labels={
                'Topic': 'Pasangan Kata Kerja-Kata Benda',
                'Frequency': 'Frekuensi'
            },
            text='Percentage'  # Menampilkan persentase di bar
        )
        
        # Update layout dan format
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='auto',
        )
        
        fig.update_layout(
            height=600,  # Menyesuaikan tinggi grafik
            yaxis={'categoryorder': 'total ascending'},  # Mengurutkan berdasarkan frekuensi
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Topic analysis data not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error displaying topic analysis: {str(e)}")

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
                st.metric('Nilai Bobot Sentimen', f"{average_sentiment_weight:.3f}")
            elif title == "Adjustment Factor 1 Open Question":
                st.metric('Nilai Bobot Sentimen', f"{average_sentiment_weight:.3f}")
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
        
        verb_active_df = analyze_action_phrases(search_df)  # Gunakan fungsi analyze_action_phrases
        
        # Tampilkan topic analysis secara terpisah
        display_topic_analysis(search_df)
        
        display_topic_analysis_ARK(search_df)
        
        display_topic_analysis_IAKD(search_df)
        display_topic_analysis_KS(search_df)
        display_topic_analysis_MS(search_df)
        display_topic_analysis_PBKN(search_df)
        display_topic_analysis_PEPK(search_df)
        display_topic_analysis_PMDK(search_df)
        display_topic_analysis_PPDP(search_df)
        display_topic_analysis_PVML(search_df)
        
        # # Lanjutkan dengan analisis verb active
        # if not verb_active_df.empty:
        #     st.dataframe(verb_active_df, use_container_width=True)
            
        #     st.subheader('Top Words')
        #     top_words_fig = px.bar(
        #         verb_active_df.head(10), 
        #         x='keyword', 
        #         y='jumlah', 
        #         title='Top 10 Words'
        #     )
        #     st.plotly_chart(top_words_fig)
        # else:
        #     st.write("No verb or active words found in the filtered dataset.")
        
        # if keyword_filter:
        #     st.header('Keyword Analysis')
            
        #     kw_freq_data = []
        #     for kw in keyword_filter:
        #         kw_freq_row = {'Keyword': kw}
        #         for col in open_question_columns:
        #             count = filtered_df[col].apply(
        #                 lambda x: kw in extract_keywords(x, stop_words)
        #             ).sum()
        #             kw_freq_row[f'{col} Count'] = count
                
        #         kw_freq_data.append(kw_freq_row)
            
        #     kw_freq_df = pd.DataFrame(kw_freq_data)
        #     st.dataframe(kw_freq_df, use_container_width=True)
            
        #     bar_data = []
        #     for col in open_question_columns:
        #         bar_data.append(
        #             go.Bar(
        #                 name=col, 
        #                 x=kw_freq_df['Keyword'], 
        #                 y=kw_freq_df[f'{col} Count']
        #             )
        #         )
            
        #     fig = go.Figure(data=bar_data)
        #     fig.update_layout(barmode='group', title='Keyword Frequencies in Open Questions')
        #     st.plotly_chart(fig)
    else:
        st.warning('No data loaded. Please check the data file path.')
        
        
Path = "ojk1103"
        
def kojk_page(data):
    st.title(data)
    
    datasets1 = {
        "Gabungan": {
            'path': 'data/hasil/hasil_gabungan_KOJK.csv',
            'open_questions': ['OPEN QUESTION 1', 'OPEN QUESTION 2']
        },
        "Adjustment Factor 2 Open Question": {
            'path': 'data/hasil/main_data_28_KOJK_OQ2_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1', 'OPEN QUESTION 2']
        },
        "Adjustment Factor 1 Open Question": {
            'path': 'data/hasil/main_data_28_KOJK_OQ1_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1']
        },
        "Confirmation Factor 2 Open Question": {
            'path': 'data/hasil/main_data_28_KOJK_OQ2_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1', 'OPEN QUESTION 2']
        },
        "Confirmation Factor 1 Open Question": {
            'path': 'data/hasil/main_data_28_KOJK_OQ1_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1']
        }
    }
    
    if data in datasets1:
        df = load_data(datasets1[data]['path'])
        
        try:
            stop_words = set(stopwords.words('indonesian'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('indonesian'))
            
        if not df.empty:
            df['Title'] = data
            st.sidebar.header('Cascading Filters')
            
            filter_columns = [
                col for col in [
                    # 'JENIS SURVEI',
                    'TIPE QUESTION', 
                    # 'BIDANG',
                    'SATKER (AKRONIM)', 'Label'] 
                if col in df.columns
            ]
            
            selected_filters = {}
            filtered_df = df.copy()
            
            for i, column in enumerate(filter_columns):
                available_options = filtered_df[column].unique().tolist()
                
                filter_key = f'filter_{column}_{data}'
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
                datasets1[data]['open_questions'], 
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
                            for col in datasets1[data]['open_questions'] 
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
            
            # Survey Metrics Section
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
                st.metric('Nilai Bobot Sentimen', f"{average_sentiment_weight:.3f}")

            # Visualizations Section
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
            
            # Detailed Data Section
            st.header('Detailed Data')
            search_term = st.text_input('Search Tabel Query', placeholder="Search across all columns", key=f"search_input_{data}")

            if search_term:
                search_df = filtered_df[
                    filtered_df.apply(
                        lambda row: row.astype(str).str.contains(search_term, case=False).any(),
                        axis=1
                    )
                ]
            else:
                search_df = filtered_df

            columns_to_exclude = ['New_Label', 'Confidence', 'NAMA PIC/RESPONDEN', 'EMAIL', 'KONTAK', 'EMAIL CADANGAN', 'KOTAK CADANGAN', 'Combined_Text', 'Text', 'Title']
            display_df = search_df.drop(columns=columns_to_exclude, errors='ignore')
            
            if data == 'Confirmation Factor':
                display_df.drop(['NILAI_SENTIMEN'], axis=1, inplace=True) 
            
            if data == "Confirmation Factor 2 Open Question":
                display_df.drop(['NILAI_SENTIMEN'], axis=1, inplace=True) 
            
            if data == "Confirmation Factor 1 Open Question":
                display_df.drop(['NILAI_SENTIMEN'], axis=1, inplace=True) 
            st.dataframe(display_df)
            
            # Word Analysis Section
            display_topic_analysis_kojk(search_df)
            
        #     if not verb_active_df.empty:
        #         st.dataframe(verb_active_df, use_container_width=True)
                
        #         st.subheader('Top Words')
        #         top_words_fig = px.bar(
        #             verb_active_df.head(10), 
        #             x='keyword', 
        #             y='jumlah', 
        #             title='Top 10 Words'
        #         )
        #         st.plotly_chart(top_words_fig)
        #     else:
        #         st.write("No verb or active words found in the filtered dataset.")
            
        #     # Keyword Analysis Section
        #     if keyword_filter:
        #         st.header('Keyword Analysis')
                
        #         kw_freq_data = []
        #         for kw in keyword_filter:
        #             kw_freq_row = {'Keyword': kw}
        #             for col in datasets1[data]['open_questions']:
        #                 count = filtered_df[col].apply(
        #                     lambda x: kw in extract_keywords(x, stop_words)
        #                 ).sum()
        #                 kw_freq_row[f'{col} Count'] = count
                    
        #             kw_freq_data.append(kw_freq_row)
                
        #         kw_freq_df = pd.DataFrame(kw_freq_data)
        #         st.dataframe(kw_freq_df, use_container_width=True)
                
        #         bar_data = []
        #         for col in datasets1[data]['open_questions']:
        #             bar_data.append(
        #                 go.Bar(
        #                     name=col, 
        #                     x=kw_freq_df['Keyword'], 
        #                     y=kw_freq_df[f'{col} Count']
        #                 )
        #             )
                
        #         fig = go.Figure(data=bar_data)
        #         fig.update_layout(barmode='group', title='Keyword Frequencies in Open Questions')
        #         st.plotly_chart(fig)
        else:
            st.warning('No data loaded. Please check the data file path.')
            
def idi_page(data):
    st.title(data)
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
        
        if data == 'Confirmation Factor':
            average_sentiment_weight_idi = calculate_sentiment_weight(
                idi_df, 
                filter_columns,
                True
            )
        else:
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
                if data == 'Confirmation Factor':
                    st.metric('Nilai Bobot Sentimen', f"{(average_sentiment_weight_idi):.2f}")
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
        
        if data == 'Confirmation Factor':
            display_df.drop(['NILAI_SENTIMEN'], axis=1, inplace=True) 
        st.dataframe(display_df)

        display_topic_analysis_idi(search_df)
        
        # if not verb_active_df.empty:
        #     st.dataframe(verb_active_df, use_container_width=True)
            
        #     st.subheader('Top Words')
        #     top_words_fig = px.bar(
        #         verb_active_df.head(10), 
        #         x='keyword', 
        #         y='jumlah', 
        #         title='Top 10 Words'
        #     )
        #     st.plotly_chart(top_words_fig)
        # else:
        #     st.write("No verb or active words found in the filtered dataset.")

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
    "IDI Dashboard": idi_page,
    "KOJK Dashboard": kojk_page
}

def login():
    st.markdown("""
        <style>
        @media screen and (max-width: 640px) {
            .hide-mobile {
                display: none !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
     
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
            <div style='padding: 20px; 
                        border-radius: 10px;
                        height: 350px;
                        max-hight: 100vh;
                        margin-top: 50px;
                        display: flex;
                        align-items: center;
                        justify-content: center;' class="hide-mobile">
                        <img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAvMAAAFLCAYAAABbdANSAAAAAXNSR0IArs4c6QAAIABJREFUeF7svQm8XVV59/+sYU9numPuzTyRBElIEMIkM0URcayWSYtTFapYFGWwfdv/i/+2aofXtgr2xdcqgkALVlsHEFFGDUOCJAxhSELm3Hk6057W8L7PPvcmNyEJISHAJc/+wOfk3HP2Pmt/1zr7/Nazf+t5GNBGBIgAESACRIAIEAEiQASIwIQkwCZkq6nRRIAIEAEiQASIABEgAkSACACJeRoERIAIEAEiQASIABEgAkRgghIgMT9BO46aTQSIABEgAkSACBABIkAESMzTGCACRIAIEAEiQASIABEgAhOUAIn5Cdpx1GwiQASIABEgAkSACBABIkBinsYAESACRIAIEAEiQASIABGYoARIzE/QjqNmEwEiQASIABEgAkSACBABEvM0BogAESACRIAIEAEiQASIwAQlQGJ+gnYcNZsIEAEiQASIABEgAkSACJCYpzFABIgAESACRIAIEAEiQAQmKAES8xO046jZRIAIEAEiQASIABEgAkSAxDyNASJABIgAESACRIAIEAEiMEEJkJifoB1HzSYCRIAIEAEiQASIABEgAiTmaQwQASJABIgAESACRIAIEIEJSoDE/ATtOGo2ESACRIAIEAEiQASIABEgMU9jgAgQASJABIgAESACRIAITFACJOYnaMdRs4kAESACRIAIEAEiQASIAIl5GgNEgAgQASJABIgAESACRGCCEiAxP0E7jppNBIgAESACRIAIEAEiQARIzNMYIAJEgAgQASJABIgAESACE5QAifkJ2nHUbCJABIgAESACRIAIEAEiQGKexgARIAJEgAgQASJABIgAEZigBEjMT9COo2YTASJABIgAESACRIAIEAES8zQGiAARIAJEgAgQASJABIjABCVAYn6Cdhw1mwgQASJABIgAESACRIAIkJinMUAEiAARIAJEgAgQASJABCYoARLzE7TjqNlEgAgQASJABIgAESACRIDEPI0BIkAEiAARIAJEgAgQASIwQQmQmJ+gHUfNJgJEgAgQASJABIgAESACJOZpDBABIkAEiAARIAJEgAgQgQlKgMT8BO04ajYRIAJEgAgQASJABIgAESAxT2OACBABIkAEiAARIAJEgAhMUAIk5idox1GziQARIAJEgAgQASJABIgAiXkaA0SACBABIkAEiAARIAJEYIISIDE/QTuOmk0EiAARIAJEgAgQASJABEjM0xggAkSACBABIkAEiAARIAITlACJ+QnacdRsIkAEiAARIAJEgAgQASJAYp7GABEgAkSACBABIkAEiAARmKAESMxP0I6jZhMBIkAEiAARIAJEgAgQARLzNAaIABEgAkSACBABIkAEiMAEJUBifoJ2HDWbCBABIkAEiAARIAJEgAiQmKcxQASIABEgAkSACBABIkAEJigBEvMTtOOo2USACBABIkAEiAARIAJEgMQ8jQEiQASIABEgAkSACBABIjBBCZCYn6AdR80mAkSACBABIkAEiAARIAIk5mkMEAEiQASIABEgAkSACBCBCUqAxPwE7ThqNhEgAkSACBABIkAEiAARIDFPY4AIEAEiQASIABEgAkSACExQAiTmJ2jHUbOJABEgAkSACBABIkAEiACJeRoDRIAIEAEiQASIABEgAkRgghIgMT9BO46aTQSIABEgAkSACBABIkAESMzTGCACRIAIEAEiQASIABEgAhOUAIn5Cdpx1GwiQASIABEgAkSACBABIkBinsYAESACRIAIEAEiQASIABGYoARIzE/QjqNmEwEiQASIABEgAkSACBABEvM0BogAESACRIAIEAEiQASIwAQlQGJ+gnYcNZsIEAEiQASIABEgAkSACJCYpzFABIgAESACRIAIEAEiQAQmKAES8xO046jZRIAIEAEiQASIABEgAkSAxDyNASJABIgAESACRIAIEAEiMEEJkJifoB1HzSYCRIAIEAEiQASIABEgAiTmaQwQASJABIgAESACRIAIEIEJSoDE/ATtOGo2ESACRIAIEAEiQASIABEgMU9jgAgQASJABIgAESACRIAITFACJOYnaMdRs4kAESACRIAIEAEiQASIAIl5GgNEgAgQASJABIgAESACRGCCEiAxP0E7jppNBIgAESACRIAIEAEiQARIzNMYIAJEgAgQASJABIgAESACE5QAifkJ2nHUbCJABIgAESACRIAIEAEicEiLeWstfxxA0DAgAkTg1SHwd3fcwardBfaJP3uXOp8x/eoclY5CBIgAESACRIAI7InAISvmUchf/C93zK/mm49JWCAN55ZxbvHRGLDCWmsAgDPWYDT6HB+1AGBWW/yzsTx7xBkBKpexR24YwyfcAMP3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' style='max-width: 100%; 
                                        max-height: 100%; 
                                        object-fit: contain;
                                        border-radius: 5px;'>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='padding: 20px; 
                        margin-top: 50px;
                        border-radius: 10px;
                        max-hight: 100vh;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h2 style='text-align: center;'>Login</h2>
                <p style='text-align: center'>Survey Kepuasan Stakeholder OJK 2024</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col2_1, col2_2, col2_3 = st.columns([1,1,1])
        with col2_2:
            if st.button("Login", use_container_width=True):
                if username == Path and password == Title:
                    st.session_state['logged_in'] = True
                else:
                    st.error("Invalid username or password")
                    
def main():
    try:
        stop_words = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('indonesian'))
    
    datasets = {
        "Gabungan": {
            'path': 'data/hasil/hasil_gabungan.csv',
            'open_questions': ['OPEN QUESTION 1', 'OPEN QUESTION 2']
        },
        "Adjustment Factor 2 Open Question": {
            'path': 'data/hasil/main_data_28_OQ2_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1', 'OPEN QUESTION 2']
        },
        "Adjustment Factor 1 Open Question": {
            'path': 'data/hasil/main_data_28_OQ1_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1']
        },
        "Confirmation Factor 2 Open Question": {
            'path': 'data/hasil/main_data_28_OQ2_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1', 'OPEN QUESTION 2']
        },
        "Confirmation Factor 1 Open Question": {
            'path': 'data/hasil/main_data_28_OQ1_dashboard.csv',
            'open_questions': ['OPEN QUESTION 1']
        }
    }

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login()
    else:
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Select a Page", list(pages.keys()))
        
        if page == "IDI Dashboard":
            data = st.sidebar.selectbox("Select a Dataset", ['Adjusment Factor', 'Confirmation Factor'])
            pages[page](data)
        elif page == "KOJK Dashboard":
            data1 = st.sidebar.selectbox("Select a Dataset", list(datasets.keys()))
            pages[page](data1)
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
            
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False

if __name__ == '__main__':
    main()