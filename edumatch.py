import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from io import BytesIO
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="EduMatch Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4B5563;
    }
    .info-text {
        font-size: 0.9rem;
        color: #4B5563;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'students_df' not in st.session_state:
    st.session_state.students_df = None
if 'academic_df' not in st.session_state:
    st.session_state.academic_df = None
if 'softskills_df' not in st.session_state:
    st.session_state.softskills_df = None
if 'interests_df' not in st.session_state:
    st.session_state.interests_df = None
if 'extracurricular_df' not in st.session_state:
    st.session_state.extracurricular_df = None
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = None

# Define category mappings and constants
SUBJECT_CATEGORIES = {
    'Sains': ['Matematika', 'Fisika', 'Kimia', 'Biologi'],
    'Sosial': ['Sejarah', 'Ekonomi', 'Geografi', 'Sosiologi'],
    'Bahasa': ['B. Indonesia', 'B. Inggris'],
    'Lainnya': ['Seni', 'Olahraga', 'Agama', 'PKN']
}

SOFT_SKILLS = [
    'Komunikasi', 'Kepemimpinan', 'Pemecahan Masalah',
    'Kreativitas', 'Kerja Sama Tim', 'Adaptabilitas',
    'Manajemen Waktu', 'Kemandirian'
]

HOLLAND_INTERESTS = [
    'Realistic', 'Investigative', 'Artistic',
    'Social', 'Enterprising', 'Conventional'
]

EXTRACURRICULAR_CATEGORIES = [
    'Sains & Teknologi', 'Sosial & Organisasi', 'Olahraga',
    'Seni & Budaya', 'Bahasa', 'Keagamaan'
]

JURUSAN_CATEGORIES = [
    'STEM', 'Kedokteran & Kesehatan', 'Bisnis & Ekonomi',
    'Ilmu Sosial & Humaniora', 'Seni & Desain', 'Pendidikan'
]

# Define cluster descriptions for interpretation
CLUSTER_DESCRIPTIONS = {
    0: {
        'name': 'Academically Strong with Technical Skills',
        'description': 'Siswa dalam kelompok ini memiliki nilai akademik tinggi, terutama di bidang sains, '
                      'dengan kekuatan di pemecahan masalah dan orientasi investigative. Mereka aktif dalam '
                      'kegiatan ekstrakurikuler sains dan teknologi.',
        'recommendations': ['STEM', 'Kedokteran', 'Teknik']
    },
    1: {
        'name': 'Creative-Practical Learners',
        'description': 'Siswa dalam kelompok ini memiliki nilai akademik sedang-rendah, namun unggul dalam '
                      'kreativitas dan adaptabilitas. Mereka sangat aktif dalam kegiatan ekstrakurikuler '
                      'terutama seni dan olahraga.',
        'recommendations': ['Seni & Desain', 'Kinesiology', 'Teknik Terapan']
    },
    2: {
        'name': 'Socially Competent Communicators',
        'description': 'Siswa dalam kelompok ini memiliki nilai akademik menengah, dengan kekuatan di bidang '
                      'bahasa dan sosial. Soft skills mereka sangat tinggi terutama di komunikasi dan kerja '
                      'sama tim. Orientasi minat mereka dominan di bidang sosial dan enterprising.',
        'recommendations': ['Ilmu Sosial & Humaniora', 'Bisnis', 'Pendidikan']
    },
    3: {
        'name': 'Underperforming with Growth Potential',
        'description': 'Siswa dalam kelompok ini menunjukkan nilai di bawah rata-rata pada beberapa aspek. '
                      'Mereka membutuhkan program pengembangan yang terarah untuk mengidentifikasi dan '
                      'mengembangkan potensi mereka.',
        'recommendations': ['Program Pengembangan Soft Skills', 'Eksplorasi Minat', 'Motivasi Akademik']
    }
}

# Define scoring matrices
# This matrix defines how important each factor is for different degree categories
# Values range from 0 (not important) to 1 (very important)
SCORING_MATRIX = {
    'STEM': {
        'Sains': 0.5,
        'Sosial': 0.1, 
        'Bahasa': 0.1,
        'Lainnya': 0.05,
        'Pemecahan Masalah': 0.15,
        'Investigative': 0.25,
        'Ekskul Sains & Teknologi': 0.1
    },
    'Kedokteran & Kesehatan': {
        'Sains': 0.6,
        'Sosial': 0.1,
        'Bahasa': 0.1,
        'Lainnya': 0.05,
        'Kerja Sama Tim': 0.1,
        'Social': 0.15,
        'Investigative': 0.1,
        'Ekskul Sosial & Organisasi': 0.05
    },
    'Bisnis & Ekonomi': {
        'Sains': 0.2,
        'Sosial': 0.3,
        'Bahasa': 0.2,
        'Lainnya': 0.05,
        'Kepemimpinan': 0.15,
        'Enterprising': 0.2,
        'Conventional': 0.1,
        'Ekskul Sosial & Organisasi': 0.05
    },
    'Ilmu Sosial & Humaniora': {
        'Sains': 0.1,
        'Sosial': 0.25,
        'Bahasa': 0.3,
        'Lainnya': 0.05,
        'Komunikasi': 0.15,
        'Social': 0.2,
        'Artistic': 0.05,
        'Ekskul Sosial & Organisasi': 0.05
    },
    'Seni & Desain': {
        'Sains': 0.1,
        'Sosial': 0.1,
        'Bahasa': 0.2,
        'Lainnya': 0.2,
        'Kreativitas': 0.25,
        'Artistic': 0.3,
        'Ekskul Seni & Budaya': 0.1
    },
    'Pendidikan': {
        'Sains': 0.15,
        'Sosial': 0.2,
        'Bahasa': 0.25,
        'Lainnya': 0.1,
        'Komunikasi': 0.2,
        'Social': 0.25,
        'Ekskul Sosial & Organisasi': 0.05
    }
}

# Weights for different components
COMPONENT_WEIGHTS = {
    'Akademik': 0.4,
    'Soft Skills': 0.3,
    'Minat Bakat': 0.2,
    'Ekstrakurikuler': 0.1
}

# Create template dataframes for downloads
def create_template_dataframes():
    # Student basic data template
    students_template = pd.DataFrame({
        'student_id': ['S001', 'S002'],
        'nama': ['John Doe', 'Jane Smith'],
        'kelas': ['11 IPA 1', '11 IPA 2'],
        'jenis_kelamin': ['L', 'P'],
        'tanggal_lahir': ['2006-05-15', '2006-08-21']
    })
    
    # Academic scores template
    subjects = []
    for category in SUBJECT_CATEGORIES:
        subjects.extend(SUBJECT_CATEGORIES[category])
    
    academic_template = pd.DataFrame({
        'student_id': ['S001', 'S002']
    })
    for subject in subjects:
        academic_template[subject] = [85, 78]
    
    # Soft skills template
    softskills_template = pd.DataFrame({
        'student_id': ['S001', 'S002']
    })
    for skill in SOFT_SKILLS:
        softskills_template[skill] = [4, 3]
    
    # Interests template
    interests_template = pd.DataFrame({
        'student_id': ['S001', 'S002']
    })
    for interest in HOLLAND_INTERESTS:
        interests_template[interest] = [3, 4]
    
    # Extracurricular template
    extracurricular_template = pd.DataFrame({
        'student_id': ['S001', 'S002'],
        'jenis_ekstrakurikuler': ['Robotik', 'OSIS'],
        'kategori': ['Sains & Teknologi', 'Sosial & Organisasi'],
        'tingkat_keterlibatan': [5, 4],
        'prestasi': ['Ya', 'Tidak']
    })
    
    return {
        'students': students_template,
        'academic': academic_template,
        'softskills': softskills_template,
        'interests': interests_template,
        'extracurricular': extracurricular_template
    }

# Function to download dataframe as Excel
def download_excel(df, filename):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.close()
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Function to create single Excel file with multiple sheets
def create_multi_sheet_excel(dfs_dict):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    for sheet_name, df in dfs_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    writer.close()
    output.seek(0)
    
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="EduMatch_Templates.xlsx">Download All Templates (Excel)</a>'
    return href

# Data preprocessing functions
def preprocess_data():
    if (st.session_state.students_df is None or st.session_state.academic_df is None or 
        st.session_state.softskills_df is None or st.session_state.interests_df is None or 
        st.session_state.extracurricular_df is None):
        return None, None, None
    
    # Merge all dataframes on student_id
    merged_df = st.session_state.students_df.copy()
    
    # Aggregate academic scores by category
    academic_categories = {}
    for category, subjects in SUBJECT_CATEGORIES.items():
        # Only use subjects that exist in the dataframe
        valid_subjects = [s for s in subjects if s in st.session_state.academic_df.columns]
        if valid_subjects:
            academic_categories[category] = st.session_state.academic_df[valid_subjects].mean(axis=1)
    
    academic_cat_df = pd.DataFrame(academic_categories)
    academic_cat_df['student_id'] = st.session_state.academic_df['student_id']
    
    # Aggregate soft skills (convert to 0-100 scale)
    softskills_df = st.session_state.softskills_df.copy()
    # Convert 1-5 scale to 0-100
    for skill in SOFT_SKILLS:
        if skill in softskills_df.columns:
            softskills_df[skill] = (softskills_df[skill] - 1) * 25
            
    # Aggregate interests (convert to 0-100 scale)
    interests_df = st.session_state.interests_df.copy()
    # Convert 1-5 scale to 0-100
    for interest in HOLLAND_INTERESTS:
        if interest in interests_df.columns:
            interests_df[interest] = (interests_df[interest] - 1) * 25
    
    # Process extracurricular data
    # Create a wide format where each category is a column
    extracurricular_wide = pd.DataFrame({'student_id': merged_df['student_id'].unique()})
    
    for category in EXTRACURRICULAR_CATEGORIES:
        # Filter extracurricular activities for this category
        category_activities = st.session_state.extracurricular_df[
            st.session_state.extracurricular_df['kategori'] == category
        ]
        
        if not category_activities.empty:
            # Calculate engagement score (0-100)
            # Base score from involvement level (1-5 → 0-100)
            category_activities['score'] = (category_activities['tingkat_keterlibatan'] - 1) * 25
            
            # Add bonus for achievements (if yes, add 20 points, max 100)
            category_activities.loc[category_activities['prestasi'] == 'Ya', 'score'] += 20
            category_activities['score'] = category_activities['score'].clip(upper=100)
            
            # Get the max score for each student in this category
            category_scores = category_activities.groupby('student_id')['score'].max().reset_index()
            
            # Merge with wide format
            extracurricular_wide = extracurricular_wide.merge(
                category_scores,
                on='student_id',
                how='left'
            )
            
            # Rename the column
            extracurricular_wide.rename(columns={'score': f'Ekskul {category}'}, inplace=True)
            
    # Fill NaN values with 0 (no participation in that category)
    extracurricular_wide.fillna(0, inplace=True)
    
    # Merge all datasets
    analysis_df = merged_df.merge(academic_cat_df, on='student_id', how='left')
    
    # Calculate average soft skills
    softskills_avg = softskills_df[SOFT_SKILLS].mean(axis=1)
    softskills_df['Soft Skills Avg'] = softskills_avg
    
    # Calculate average interests
    interests_avg = interests_df[HOLLAND_INTERESTS].mean(axis=1)
    interests_df['Interests Avg'] = interests_avg
    
    # Merge with soft skills and interests
    for skill in SOFT_SKILLS:
        if skill in softskills_df.columns:
            analysis_df = analysis_df.merge(
                softskills_df[['student_id', skill]],
                on='student_id',
                how='left'
            )
    
    for interest in HOLLAND_INTERESTS:
        if interest in interests_df.columns:
            analysis_df = analysis_df.merge(
                interests_df[['student_id', interest]],
                on='student_id',
                how='left'
            )
    
    # Merge with extracurricular
    analysis_df = analysis_df.merge(extracurricular_wide, on='student_id', how='left')
    
    # Fill NaN values
    analysis_df.fillna(0, inplace=True)
    
    # Create feature matrix for clustering
    # Select only numerical columns for clustering
    numerical_cols = (
        list(academic_categories.keys()) + 
        SOFT_SKILLS + 
        HOLLAND_INTERESTS + 
        [f'Ekskul {cat}' for cat in EXTRACURRICULAR_CATEGORIES]
    )
    
    # Filter only columns that exist in the dataframe
    numerical_cols = [col for col in numerical_cols if col in analysis_df.columns]
    
    if not numerical_cols:
        st.error("No numerical columns found for analysis")
        return None, None, None
    
    # Create feature matrix
    X = analysis_df[numerical_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return analysis_df, X_scaled, numerical_cols

# Clustering functions
def perform_clustering(X, n_clusters=4):
    if X is None:
        return None, None
    
    # Create linkage matrix
    linkage_matrix = linkage(X, method='ward')
    
    # Perform hierarchical clustering
    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )
    
    clusters = hc.fit_predict(X)
    
    return clusters, linkage_matrix

# Generate recommendations
def generate_recommendations(analysis_df, clusters):
    if analysis_df is None or clusters is None:
        return None
    
    # Add cluster assignments to dataframe
    analysis_df['cluster'] = clusters
    
    # Calculate scores for each jurusan category
    recommendations = pd.DataFrame({'student_id': analysis_df['student_id']})
    
    # Calculate scores for each student and each jurusan
    for jurusan in JURUSAN_CATEGORIES:
        # Initialize score column
        jurusan_score = np.zeros(len(analysis_df))
        
        # Apply the scoring matrix
        for factor, weight in SCORING_MATRIX[jurusan].items():
            if factor in analysis_df.columns:
                # Academic categories and extracurricular are already 0-100 scaled
                jurusan_score += analysis_df[factor] * weight
            elif factor in SOFT_SKILLS:
                # Soft skills should be in the dataframe already 0-100 scaled
                if factor in analysis_df.columns:
                    jurusan_score += analysis_df[factor] * weight
            elif factor in HOLLAND_INTERESTS:
                # Interests should be in the dataframe already 0-100 scaled
                if factor in analysis_df.columns:
                    jurusan_score += analysis_df[factor] * weight
        
        # Add jurusan score to recommendations dataframe
        recommendations[jurusan] = jurusan_score
    
    # Add cluster assignments to recommendations
    recommendations['cluster'] = analysis_df['cluster']
    
    # Add student name
    recommendations = recommendations.merge(
        analysis_df[['student_id', 'nama']],
        on='student_id',
        how='left'
    )
    
    return recommendations

# Visualization functions
def plot_dendrogram(linkage_matrix):
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linkage_matrix, ax=ax)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or cluster size')
    plt.ylabel('Distance')
    return fig

def plot_radar_chart(student_data, category_columns, title):
    # Prepare data for radar chart
    categories = category_columns
    values = student_data[category_columns].values.flatten().tolist()
    
    # Close the loop
    values += values[:1]
    categories += categories[:1]
    
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=student_data['nama'].values[0] if 'nama' in student_data else 'Student'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title=title
    )
    
    return fig

def plot_cluster_distribution(analysis_df, clusters):
    # Create a dataframe of cluster counts
    cluster_counts = pd.DataFrame({
        'Cluster': [CLUSTER_DESCRIPTIONS[i]['name'] for i in range(len(CLUSTER_DESCRIPTIONS))],
        'Count': [sum(clusters == i) for i in range(len(CLUSTER_DESCRIPTIONS))]
    })
    
    # Calculate percentages
    total = cluster_counts['Count'].sum()
    cluster_counts['Percentage'] = (cluster_counts['Count'] / total * 100).round(1)
    
    # Create labels
    cluster_counts['Label'] = cluster_counts.apply(
        lambda x: f"{x['Cluster']}: {x['Count']} ({x['Percentage']}%)", 
        axis=1
    )
    
    # Create pie chart
    fig = px.pie(
        cluster_counts, 
        values='Count', 
        names='Label',
        title='Distribusi Cluster Siswa',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    
    return fig

def plot_cluster_characteristics(analysis_df, clusters, numerical_cols):
    # Calculate cluster means
    cluster_means = pd.DataFrame()
    
    for i in range(len(CLUSTER_DESCRIPTIONS)):
        if i in np.unique(clusters):
            cluster_data = analysis_df[analysis_df['cluster'] == i]
            if not cluster_data.empty:
                cluster_means[f'Cluster {i}'] = cluster_data[numerical_cols].mean()
    
    # Create heatmap
    fig = px.imshow(
        cluster_means,
        labels=dict(x="Cluster", y="Feature", color="Value"),
        x=cluster_means.columns,
        y=cluster_means.index,
        color_continuous_scale="RdBu_r",
        title="Karakteristik Cluster (Nilai Rata-rata)",
        aspect="auto"
    )
    
    fig.update_layout(height=800)
    
    return fig

def plot_recommendation_chart(recommendations, student_id=None):
    if student_id:
        # Filter for a specific student
        student_recs = recommendations[recommendations['student_id'] == student_id].iloc[0]
        student_name = student_recs['nama']
        
        # Prepare data for bar chart
        jurusan_scores = []
        for jurusan in JURUSAN_CATEGORIES:
            jurusan_scores.append({
                'Jurusan': jurusan,
                'Skor': student_recs[jurusan]
            })
        
        jurusan_df = pd.DataFrame(jurusan_scores)
        
        # Sort by score (descending)
        jurusan_df = jurusan_df.sort_values('Skor', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            jurusan_df,
            x='Jurusan',
            y='Skor',
            title=f"Rekomendasi Jurusan untuk {student_name}",
            color='Skor',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(xaxis_title="Jurusan", yaxis_title="Skor Kesesuaian")
        
    else:
        # Show top recommendations for all students
        top_recs = []
        
        for _, row in recommendations.iterrows():
            # Get top 2 recommendations for this student
            top_jurusan = [(jurusan, row[jurusan]) for jurusan in JURUSAN_CATEGORIES]
            top_jurusan.sort(key=lambda x: x[1], reverse=True)
            top_jurusan = top_jurusan[:2]
            
            for rank, (jurusan, score) in enumerate(top_jurusan):
                top_recs.append({
                    'Nama': row['nama'],
                    'Jurusan': jurusan,
                    'Skor': score,
                    'Peringkat': f"Rekomendasi {rank+1}"
                })
        
        top_recs_df = pd.DataFrame(top_recs)
        
        # Create grouped bar chart
        fig = px.bar(
            top_recs_df,
            x='Nama',
            y='Skor',
            color='Peringkat',
            barmode='group',
            hover_data=['Jurusan'],
            title="Top 2 Rekomendasi Jurusan per Siswa"
        )
        
        fig.update_layout(xaxis_title="Siswa", yaxis_title="Skor Kesesuaian")
        
    return fig

def plot_scatter_clusters(analysis_df, clusters, feature_x, feature_y):
    # Create a dataframe for the scatter plot
    scatter_df = pd.DataFrame({
        'student_id': analysis_df['student_id'],
        'nama': analysis_df['nama'],
        'x': analysis_df[feature_x],
        'y': analysis_df[feature_y],
        'cluster': [CLUSTER_DESCRIPTIONS[c]['name'] for c in clusters]
    })
    
    # Create scatter plot
    fig = px.scatter(
        scatter_df,
        x='x',
        y='y',
        color='cluster',
        hover_name='nama',
        title=f'Clustering Siswa: {feature_x} vs {feature_y}',
        labels={'x': feature_x, 'y': feature_y},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    return fig

def generate_student_report(student_id, analysis_df, recommendations):
    # Filter data for the selected student
    student_data = analysis_df[analysis_df['student_id'] == student_id].iloc[0]
    student_recs = recommendations[recommendations['student_id'] == student_id].iloc[0]
    
    # Get cluster information
    cluster_id = int(student_data['cluster'])
    cluster_info = CLUSTER_DESCRIPTIONS[cluster_id]
    
    # Get top 3 jurusan recommendations
    top_jurusan = [(jurusan, student_recs[jurusan]) for jurusan in JURUSAN_CATEGORIES]
    top_jurusan.sort(key=lambda x: x[1], reverse=True)
    top_jurusan = top_jurusan[:3]
    
    # Calculate strengths and areas for improvement
    strengths = []
    improvements = []
    
    # Academic strengths/improvements
    academic_scores = {}
    for category in SUBJECT_CATEGORIES:
        if category in student_data:
            academic_scores[category] = student_data[category]
    
    top_academic = sorted(academic_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    bottom_academic = sorted(academic_scores.items(), key=lambda x: x[1])[:2]
    
    for category, score in top_academic:
        if score >= 75:  # Only consider as strength if score is good
            strengths.append(f"Nilai akademik tinggi di bidang {category} ({score:.1f})")
    
    for category, score in bottom_academic:
        if score < 70:  # Only consider as improvement area if score is low
            improvements.append(f"Meningkatkan nilai akademik di bidang {category} ({score:.1f})")
    
    # Soft skills strengths/improvements
    soft_skills_scores = {}
    for skill in SOFT_SKILLS:
        if skill in student_data:
            soft_skills_scores[skill] = student_data[skill]
    
    top_skills = sorted(soft_skills_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    bottom_skills = sorted(soft_skills_scores.items(), key=lambda x: x[1])[:2]
    
    for skill, score in top_skills:
        if score >= 75:  # Only consider as strength if score is good
            strengths.append(f"Soft skill yang kuat: {skill} ({score:.1f})")
    
    for skill, score in bottom_skills:
        if score < 60:  # Only consider as improvement area if score is low
            improvements.append(f"Mengembangkan soft skill: {skill} ({score:.1f})")
    
    # Interests strengths
    interest_scores = {}
    for interest in HOLLAND_INTERESTS:
        if interest in student_data:
            interest_scores[interest] = student_data[interest]
    
    top_interests = sorted(interest_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    
    for interest, score in top_interests:
        if score >= 75:  # Only consider as strength if score is good
            strengths.append(f"Minat yang kuat di bidang {interest} ({score:.1f})")
    
    # Prepare report data
    report = {
        'student_id': student_id,
        'nama': student_data['nama'],
        'kelas': student_data['kelas'],
        'cluster_name': cluster_info['name'],
        'cluster_description': cluster_info['description'],
        'top_recommendations': top_jurusan,
        'strengths': strengths,
        'improvements': improvements
    }
    
    return report

def create_report_pdf(report, analysis_df, recommendations):
    # This function would generate a PDF report
    # For now, we'll just return the report data as HTML
    
    student_id = report['student_id']
    
    # Create recommendation bar chart
    rec_chart = plot_recommendation_chart(recommendations, student_id)
    
    # Create radar charts for academic and soft skills
    academic_cols = [col for col in SUBJECT_CATEGORIES.keys() if col in analysis_df.columns]
    softskill_cols = [col for col in SOFT_SKILLS if col in analysis_df.columns]
    interest_cols = [col for col in HOLLAND_INTERESTS if col in analysis_df.columns]
    
    student_data = analysis_df[analysis_df['student_id'] == student_id]
    
    academic_radar = plot_radar_chart(student_data, academic_cols, "Profil Akademik")
    softskill_radar = plot_radar_chart(student_data, softskill_cols, "Profil Soft Skills")
    interest_radar = plot_radar_chart(student_data, interest_cols, "Profil Minat & Bakat")
    
    # Convert the plots to HTML
    rec_chart_html = rec_chart.to_html(full_html=False, include_plotlyjs='cdn')
    academic_radar_html = academic_radar.to_html(full_html=False, include_plotlyjs='cdn')
    softskill_radar_html = softskill_radar.to_html(full_html=False, include_plotlyjs='cdn')
    interest_radar_html = interest_radar.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Generate HTML report
    html = f"""
    <html>
    <head>
        <title>Laporan Pemetaan Potensi Siswa - {report['nama']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2563EB; }}
            h2 {{ color: #3B82F6; margin-top: 30px; }}
            .infobox {{ background-color: #EFF6FF; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .recommendation {{ background-color: #ECFDF5; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .strength {{ color: #059669; }}
            .improvement {{ color: #B91C1C; }}
            .charts {{ display: flex; flex-wrap: wrap; }}
            .chart {{ width: 100%; margin-bottom: 30px; }}
            @media print {{
                .pagebreak {{ page-break-before: always; }}
            }}
        </style>
    </head>
    <body>
        <h1>Laporan Pemetaan Potensi Siswa</h1>
        
        <div class="infobox">
            <h2>Informasi Siswa</h2>
            <p><strong>Nama:</strong> {report['nama']}</p>
            <p><strong>ID Siswa:</strong> {report['student_id']}</p>
            <p><strong>Kelas:</strong> {report['kelas']}</p>
            <p><strong>Kelompok:</strong> {report['cluster_name']}</p>
        </div>
        
        <div class="infobox">
            <h2>Karakteristik Kelompok</h2>
            <p>{report['cluster_description']}</p>
        </div>
        
        <h2>Rekomendasi Jurusan</h2>
        <div class="recommendation">
            <p>Berdasarkan profil kompetensi Anda, berikut adalah rekomendasi jurusan kuliah yang sesuai:</p>
            <ol>
    """
    
    # Add top recommendations
    for i, (jurusan, score) in enumerate(report['top_recommendations']):
        if i < 3:  # Only show top 3
            html += f"<li><strong>{jurusan}</strong> (Skor: {score:.1f})</li>"
    
    html += """
            </ol>
        </div>
        
        <div class="chart">
    """
    
    # Add recommendation chart
    html += rec_chart_html
    
    html += """
        </div>
        
        <div class="pagebreak"></div>
        
        <h2>Profil Kompetensi</h2>
        <div class="charts">
            <div class="chart">
    """
    
    # Add academic radar chart
    html += academic_radar_html
    
    html += """
            </div>
            <div class="chart">
    """
    
    # Add soft skills radar chart
    html += softskill_radar_html
    
    html += """
            </div>
            <div class="chart">
    """
    
    # Add interests radar chart
    html += interest_radar_html
    
    html += """
            </div>
        </div>
        
        <h2>Kekuatan dan Area Pengembangan</h2>
        <div class="infobox">
            <h3>Kekuatan</h3>
            <ul>
    """
    
    # Add strengths
    for strength in report['strengths']:
        html += f"<li class='strength'>{strength}</li>"
    
    html += """
            </ul>
            
            <h3>Area Pengembangan</h3>
            <ul>
    """
    
    # Add improvement areas
    for improvement in report['improvements']:
        html += f"<li class='improvement'>{improvement}</li>"
    
    html += """
            </ul>
        </div>
        
        <div class="infobox">
            <h2>Saran Pengembangan</h2>
            <p>Berikut adalah saran untuk mengembangkan potensi Anda sesuai dengan minat dan pilihan jurusan:</p>
            <ul>
    """
    
    # Add development suggestions based on top recommendation
    top_jurusan = report['top_recommendations'][0][0]
    
    if top_jurusan == 'STEM':
        html += """
                <li>Ikuti kompetisi sains dan matematika untuk mengasah kemampuan analitis</li>
                <li>Bergabung dengan klub robotik atau pemrograman di sekolah</li>
                <li>Kembangkan kemampuan pemecahan masalah melalui latihan soal yang menantang</li>
                <li>Eksplorasi topik sains terapan melalui proyek penelitian sederhana</li>
        """
    elif top_jurusan == 'Kedokteran & Kesehatan':
        html += """
                <li>Ikuti kegiatan sosial kesehatan seperti PMR atau bakti sosial kesehatan</li>
                <li>Kembangkan kemampuan kerja sama tim melalui aktivitas kelompok</li>
                <li>Pelajari topik biologi dan kimia secara mendalam</li>
                <li>Latih kemampuan empati dan komunikasi interpersonal</li>
        """
    elif top_jurusan == 'Bisnis & Ekonomi':
        html += """
                <li>Bergabung dengan klub kewirausahaan atau ekonomi di sekolah</li>
                <li>Kembangkan kemampuan kepemimpinan melalui organisasi siswa</li>
                <li>Latih kemampuan presentasi dan komunikasi publik</li>
                <li>Ikuti seminar atau workshop tentang kewirausahaan dan ekonomi</li>
        """
    elif top_jurusan == 'Ilmu Sosial & Humaniora':
        html += """
                <li>Bergabung dengan klub debat atau jurnalistik</li>
                <li>Kembangkan kemampuan menulis melalui blog atau majalah sekolah</li>
                <li>Ikuti kompetisi esai atau karya tulis ilmiah</li>
                <li>Latih kemampuan analisis sosial melalui penelitian sederhana</li>
        """
    elif top_jurusan == 'Seni & Desain':
        html += """
                <li>Kembangkan portofolio karya seni atau desain</li>
                <li>Ikuti workshop atau kursus teknik seni dan desain</li>
                <li>Bergabung dengan klub seni atau teater di sekolah</li>
                <li>Latih kemampuan kreativitas melalui proyek seni mandiri</li>
        """
    elif top_jurusan == 'Pendidikan':
        html += """
                <li>Ikuti kegiatan mengajar sukarela atau menjadi tutor sebaya</li>
                <li>Kembangkan kemampuan komunikasi dan presentasi</li>
                <li>Bergabung dengan klub atau organisasi yang berfokus pada pengembangan pendidikan</li>
                <li>Latih kesabaran dan empati dalam berinteraksi dengan berbagai tipe kepribadian</li>
        """
    
    html += """
            </ul>
        </div>
        
        <div class="infobox">
            <h2>Catatan untuk Orang Tua</h2>
            <p>Laporan ini memberikan gambaran tentang profil kompetensi dan potensi putra/putri Anda. Beberapa hal yang dapat Anda lakukan untuk mendukung perkembangan putra/putri Anda:</p>
            <ul>
                <li>Diskusikan hasil pemetaan ini secara terbuka dengan putra/putri Anda</li>
                <li>Berikan dukungan untuk mengeksplorasi minat dan potensi sesuai rekomendasi</li>
                <li>Fasilitasi kegiatan pengembangan diri yang sesuai dengan minat dan potensi</li>
                <li>Konsultasikan dengan guru BK untuk mendapatkan panduan lebih lanjut</li>
            </ul>
        </div>
        
        <p style="margin-top: 50px; text-align: center; font-size: 12px; color: #6B7280;">
            Laporan ini dibuat oleh Sistem EduMatch. Tanggal: {datetime.now().strftime('%d-%m-%Y')}
        </p>
        
    </body>
    </html>
    """
    
    return html

# ------ MAIN APP ------

def main():
    # App title and description
    st.markdown("<h1 class='main-header'>EduMatch Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align:center; font-size:1.2rem; margin-bottom:2rem;'>
        Sistem Pemetaan Potensi dan Rekomendasi Jurusan untuk Siswa SMA/SMK
    </p>
    """, unsafe_allow_html=True)
    
    # Create navigation tabs
    tabs = st.tabs([
        "🏠 Beranda",
        "📊 Dashboard Overview",
        "👤 Analisis Individual",
        "👥 Analisis Kelompok",
        "🎯 Sistem Rekomendasi",
        "🗃️ Manajemen Data"
    ])
    
    # ===== HOME TAB =====
    with tabs[0]:
        st.markdown("<h2 class='sub-header'>Selamat Datang di EduMatch</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        EduMatch adalah platform berbasis data untuk membantu guru BK dalam melakukan pemetaan potensi
        siswa dan memberikan rekomendasi jurusan kuliah yang sesuai dengan profil kompetensi siswa secara holistik.
        """)
        
        st.markdown("<h3 class='section-header'>Tujuan</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        1. Memberikan alat analisis data komprehensif untuk guru BK dalam pemetaan potensi siswa
        2. Membantu siswa memahami kekuatan dan area pengembangan diri mereka
        3. Menghasilkan rekomendasi jurusan kuliah berdasarkan profil kompetensi yang holistik
        4. Menyediakan visualisasi data yang intuitif untuk memudahkan interpretasi hasil analisis
        """)
        
        st.markdown("<h3 class='section-header'>Komponen Penilaian</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>40%</div>
                <div class='metric-label'>Akademik</div>
                <p class='info-text'>Nilai mata pelajaran dari rapor siswa (1-100)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>30%</div>
                <div class='metric-label'>Soft Skills</div>
                <p class='info-text'>Penilaian diri untuk 8 dimensi soft skills (1-5)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>20%</div>
                <div class='metric-label'>Minat & Bakat</div>
                <p class='info-text'>Model RIASEC Holland Code (1-5)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>10%</div>
                <div class='metric-label'>Ekstrakurikuler</div>
                <p class='info-text'>Keterlibatan dan prestasi (1-5)</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<h3 class='section-header'>Cara Menggunakan Dashboard</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        1. **Persiapan Data**: Unduh template Excel dari tab Manajemen Data dan isi dengan data siswa
        2. **Upload Data**: Upload file Excel/CSV yang telah diisi di tab Manajemen Data
        3. **Analisis Overview**: Lihat statistik agregat dan distribusi siswa di tab Dashboard Overview
        4. **Analisis Individual**: Pilih siswa untuk melihat profil detail dan rekomendasi jurusan
        5. **Analisis Kelompok**: Eksplorasi hasil clustering dan karakteristik kelompok
        6. **Laporan**: Unduh laporan rekomendasi individu dari tab Analisis Individual
        """)
        
        # Show status if data is uploaded
        if st.session_state.data_uploaded:
            st.success("✅ Data telah diunggah dan siap dianalisis. Silakan jelajahi tab-tab lainnya.")
        else:
            st.warning("⚠️ Belum ada data yang diunggah. Silakan unduh template dan upload data di tab Manajemen Data.")
            
            # Show sample dashboard
            st.markdown("<h3 class='section-header'>Preview Dashboard</h3>", unsafe_allow_html=True)
            st.image("https://via.placeholder.com/800x400?text=EduMatch+Dashboard+Preview", use_column_width=True)
    
    # ===== DASHBOARD OVERVIEW TAB =====
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Dashboard Overview</h2>", unsafe_allow_html=True)
        
        if not st.session_state.data_uploaded:
            st.warning("⚠️ Belum ada data yang diunggah. Silakan upload data di tab Manajemen Data.")
        else:
            analysis_df, X_scaled, numerical_cols = preprocess_data()
            
            if analysis_df is not None:
                # Summary statistics
                st.markdown("<h3 class='section-header'>Statistik Umum</h3>", unsafe_allow_html=True)
                
                num_students = len(analysis_df)
                num_classes = len(analysis_df['kelas'].unique())
                gender_counts = analysis_df['jenis_kelamin'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{num_students}</div>
                        <div class='metric-label'>Total Siswa</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{num_classes}</div>
                        <div class='metric-label'>Jumlah Kelas</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{gender_counts.get('L', 0)} / {gender_counts.get('P', 0)}</div>
                        <div class='metric-label'>Laki-laki / Perempuan</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show cluster distribution if available
                if st.session_state.cluster_results is not None:
                    st.markdown("<h3 class='section-header'>Distribusi Kelompok Siswa</h3>", unsafe_allow_html=True)
                    
                    # Plot cluster distribution
                    clusters = st.session_state.cluster_results
                    cluster_dist_fig = plot_cluster_distribution(analysis_df, clusters)
                    st.plotly_chart(cluster_dist_fig, use_container_width=True)
                
                # Academic performance distribution
                st.markdown("<h3 class='section-header'>Distribusi Nilai Akademik</h3>", unsafe_allow_html=True)
                
                # Get academic columns
                academic_cols = [col for col in SUBJECT_CATEGORIES.keys() if col in analysis_df.columns]
                
                if academic_cols:
                    # Create a dataframe for plotting
                    academic_melt = pd.melt(
                        analysis_df,
                        id_vars=['student_id', 'nama'],
                        value_vars=academic_cols,
                        var_name='Kategori',
                        value_name='Nilai'
                    )
                    
                    # Plot histogram with KDE
                    hist_fig = px.histogram(
                        academic_melt,
                        x='Nilai',
                        color='Kategori',
                        marginal='violin',
                        opacity=0.7,
                        barmode='overlay',
                        title='Distribusi Nilai Akademik per Kategori'
                    )
                    
                    st.plotly_chart(hist_fig, use_container_width=True)
                
                # Soft skills distribution
                st.markdown("<h3 class='section-header'>Distribusi Soft Skills</h3>", unsafe_allow_html=True)
                
                # Get soft skills columns
                softskill_cols = [col for col in SOFT_SKILLS if col in analysis_df.columns]
                
                if softskill_cols:
                    # Create box plot
                    softskill_melt = pd.melt(
                        analysis_df,
                        id_vars=['student_id', 'nama'],
                        value_vars=softskill_cols,
                        var_name='Soft Skill',
                        value_name='Nilai'
                    )
                    
                    box_fig = px.box(
                        softskill_melt,
                        x='Soft Skill',
                        y='Nilai',
                        title='Distribusi Nilai Soft Skills'
                    )
                    
                    st.plotly_chart(box_fig, use_container_width=True)
                
                # Correlation heatmap
                st.markdown("<h3 class='section-header'>Korelasi Antar Variabel</h3>", unsafe_allow_html=True)
                
                # Select columns for correlation
                corr_cols = []
                
                # Add academic columns
                corr_cols.extend([col for col in SUBJECT_CATEGORIES.keys() if col in analysis_df.columns])
                
                # Add important soft skills
                important_skills = ['Komunikasi', 'Pemecahan Masalah', 'Kreativitas', 'Kerja Sama Tim']
                corr_cols.extend([col for col in important_skills if col in analysis_df.columns])
                
                # Add important interests
                important_interests = ['Investigative', 'Social', 'Artistic']
                corr_cols.extend([col for col in important_interests if col in analysis_df.columns])
                
                if len(corr_cols) > 1:
                    # Calculate correlation matrix
                    corr_matrix = analysis_df[corr_cols].corr()
                    
                    # Create heatmap
                    corr_fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        aspect='auto',
                        color_continuous_scale='RdBu_r',
                        title='Korelasi Antar Variabel'
                    )
                    
                    st.plotly_chart(corr_fig, use_container_width=True)
    
    # ===== INDIVIDUAL ANALYSIS TAB =====
    with tabs[2]:
        st.markdown("<h2 class='sub-header'>Analisis Individual</h2>", unsafe_allow_html=True)
        
        if not st.session_state.data_uploaded:
            st.warning("⚠️ Belum ada data yang diunggah. Silakan upload data di tab Manajemen Data.")
        else:
            analysis_df, X_scaled, numerical_cols = preprocess_data()
            
            if analysis_df is not None and st.session_state.recommendations is not None:
                # Student selection
                student_options = analysis_df[['student_id', 'nama']].copy()
                student_options['display'] = student_options['student_id'] + ' - ' + student_options['nama']
                
                selected_student = st.selectbox(
                    "Pilih Siswa:",
                    options=student_options['student_id'].tolist(),
                    format_func=lambda x: student_options[student_options['student_id'] == x]['display'].iloc[0]
                )
                
                # Store selected student in session state
                st.session_state.selected_student = selected_student
                
                # Get student data
                student_data = analysis_df[analysis_df['student_id'] == selected_student].iloc[0]
                student_recs = st.session_state.recommendations[st.session_state.recommendations['student_id'] == selected_student].iloc[0]
                
                # Display student info
                st.markdown("<h3 class='section-header'>Informasi Siswa</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='card'>
                        <p><strong>Nama:</strong> {student_data['nama']}</p>
                        <p><strong>ID Siswa:</strong> {selected_student}</p>
                        <p><strong>Kelas:</strong> {student_data['kelas']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='card'>
                        <p><strong>Jenis Kelamin:</strong> {'Laki-laki' if student_data['jenis_kelamin'] == 'L' else 'Perempuan'}</p>
                        <p><strong>Tanggal Lahir:</strong> {student_data['tanggal_lahir'] if 'tanggal_lahir' in student_data else '-'}</p>
                        <p><strong>Kelompok:</strong> {CLUSTER_DESCRIPTIONS[int(student_data['cluster'])]['name']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Get top recommendation
                    top_jurusan = [(jurusan, student_recs[jurusan]) for jurusan in JURUSAN_CATEGORIES]
                    top_jurusan.sort(key=lambda x: x[1], reverse=True)
                    top_jurusan = top_jurusan[0]
                    
                    st.markdown(f"""
                    <div class='card' style='background-color: #ECFDF5;'>
                        <p><strong>Rekomendasi Jurusan Utama:</strong></p>
                        <p style='font-size: 1.3rem; font-weight: bold; color: #059669;'>{top_jurusan[0]}</p>
                        <p><strong>Skor Kesesuaian:</strong> {top_jurusan[1]:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Profile tabs
                profile_tabs = st.tabs(["Akademik", "Soft Skills", "Minat & Bakat", "Rekomendasi Jurusan"])
                
                # Academic tab
                with profile_tabs[0]:
                    st.markdown("<h4>Profil Akademik</h4>", unsafe_allow_html=True)
                    
                    # Get academic columns
                    academic_cols = [col for col in SUBJECT_CATEGORIES.keys() if col in analysis_df.columns]
                    
                    if academic_cols:
                        # Create radar chart for academic categories
                        academic_radar = plot_radar_chart(
                            student_data.to_frame().T,
                            academic_cols,
                            "Profil Akademik - Kategori"
                        )
                        
                        st.plotly_chart(academic_radar, use_container_width=True)
                    
                    # Show detailed academic scores
                    st.markdown("<h4>Detail Nilai Akademik</h4>", unsafe_allow_html=True)
                    
                    # Get all subject columns
                    subject_cols = []
                    for category in SUBJECT_CATEGORIES:
                        subject_cols.extend([s for s in SUBJECT_CATEGORIES[category] if s in st.session_state.academic_df.columns])
                    
                    if subject_cols:
                        # Get student's academic data
                        student_subjects = st.session_state.academic_df[st.session_state.academic_df['student_id'] == selected_student]
                        
                        if not student_subjects.empty:
                            # Create a dataframe for plotting
                            subjects_data = []
                            
                            for subject in subject_cols:
                                if subject in student_subjects.columns:
                                    # Get category
                                    category = next((c for c, s in SUBJECT_CATEGORIES.items() if subject in s), "Lainnya")
                                    
                                    subjects_data.append({
                                        'Mata Pelajaran': subject,
                                        'Kategori': category,
                                        'Nilai': student_subjects[subject].iloc[0]
                                    })
                            
                            subjects_df = pd.DataFrame(subjects_data)
                            
                            # Sort by category and value
                            subjects_df = subjects_df.sort_values(['Kategori', 'Nilai'], ascending=[True, False])
                            
                            # Create bar chart
                            bar_fig = px.bar(
                                subjects_df,
                                x='Mata Pelajaran',
                                y='Nilai',
                                color='Kategori',
                                title="Nilai Detail per Mata Pelajaran",
                                text='Nilai'
                            )
                            
                            bar_fig.update_layout(
                                xaxis_title="Mata Pelajaran",
                                yaxis_title="Nilai",
                                yaxis_range=[0, 100]
                            )
                            
                            st.plotly_chart(bar_fig, use_container_width=True)
                
                # Soft Skills tab
                with profile_tabs[1]:
                    st.markdown("<h4>Profil Soft Skills</h4>", unsafe_allow_html=True)
                    
                    # Get soft skills columns
                    softskill_cols = [col for col in SOFT_SKILLS if col in analysis_df.columns]
                    
                    if softskill_cols:
                        # Create radar chart for soft skills
                        softskill_radar = plot_radar_chart(
                            student_data.to_frame().T,
                            softskill_cols,
                            "Profil Soft Skills"
                        )
                        
                        st.plotly_chart(softskill_radar, use_container_width=True)
                        
                        # Show comparison with average
                        st.markdown("<h4>Perbandingan dengan Rata-rata Kelas</h4>", unsafe_allow_html=True)
                        
                        # Calculate class average
                        class_filter = analysis_df['kelas'] == student_data['kelas']
                        class_avg = analysis_df[class_filter][softskill_cols].mean()
                        
                        # Create comparison dataframe
                        comparison_data = []
                        
                        for skill in softskill_cols:
                            comparison_data.append({
                                'Soft Skill': skill,
                                'Nilai Siswa': student_data[skill],
                                'Rata-rata Kelas': class_avg[skill]
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Create comparison bar chart
                        comp_fig = go.Figure()
                        
                        comp_fig.add_trace(go.Bar(
                            x=comparison_df['Soft Skill'],
                            y=comparison_df['Nilai Siswa'],
                            name='Nilai Siswa',
                            marker_color='#3B82F6'
                        ))
                        
                        comp_fig.add_trace(go.Bar(
                            x=comparison_df['Soft Skill'],
                            y=comparison_df['Rata-rata Kelas'],
                            name='Rata-rata Kelas',
                            marker_color='#9CA3AF'
                        ))
                        
                        comp_fig.update_layout(
                            title="Perbandingan Soft Skills dengan Rata-rata Kelas",
                            xaxis_title="Soft Skill",
                            yaxis_title="Nilai",
                            barmode='group',
                            yaxis_range=[0, 100]
                        )
                        
                        st.plotly_chart(comp_fig, use_container_width=True)
                
                # Interests tab
                with profile_tabs[2]:
                    st.markdown("<h4>Profil Minat & Bakat</h4>", unsafe_allow_html=True)
                    
                    # Get interest columns
                    interest_cols = [col for col in HOLLAND_INTERESTS if col in analysis_df.columns]
                    
                    if interest_cols:
                        # Create radar chart for interests
                        interest_radar = plot_radar_chart(
                            student_data.to_frame().T,
                            interest_cols,
                            "Profil RIASEC (Holland Code)"
                        )
                        
                        st.plotly_chart(interest_radar, use_container_width=True)
                        
                        # Show RIASEC explanation
                        st.markdown("<h4>Interpretasi Profil RIASEC</h4>", unsafe_allow_html=True)
                        
                        # Get top 2 interests
                        interest_scores = {interest: student_data[interest] for interest in interest_cols if interest in student_data}
                        top_interests = sorted(interest_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                        
                        interest_descriptions = {
                            'Realistic': "Orientasi pada aktivitas praktis, fisik, dan teknikal. Cenderung menyukai bekerja dengan alat, mesin, atau hewan.",
                            'Investigative': "Orientasi pada aktivitas analitis, intelektual, dan penelitian. Menyukai memecahkan masalah abstrak dan mengeksplorasi ide.",
                            'Artistic': "Orientasi pada ekspresi diri kreatif dan tidak terstruktur. Menyukai aktivitas seni, musik, sastra, atau desain.",
                            'Social': "Orientasi pada membantu dan mengembangkan orang lain. Menyukai aktivitas mengajar, konseling, atau pelayanan.",
                            'Enterprising': "Orientasi pada memimpin dan memengaruhi orang lain. Menyukai aktivitas penjualan, manajemen, atau persuasi.",
                            'Conventional': "Orientasi pada aktivitas terstruktur dan pengelolaan data. Menyukai aktivitas yang terorganisir, detail, dan prosedural."
                        }
                        
                        # Display Holland code
                        holland_code = "".join([interest[0] for interest, _ in top_interests])
                        
                        st.markdown(f"""
                        <div class='card' style='background-color: #F5F3FF;'>
                            <p style='font-size: 1.2rem;'><strong>Holland Code Dominan:</strong> {holland_code}</p>
                            <p><strong>{top_interests[0][0]}:</strong> {interest_descriptions.get(top_interests[0][0], '')}</p>
                            <p><strong>{top_interests[1][0]}:</strong> {interest_descriptions.get(top_interests[1][0], '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show extracurricular info if available
                        st.markdown("<h4>Kegiatan Ekstrakurikuler</h4>", unsafe_allow_html=True)
                        
                        extracurricular_data = st.session_state.extracurricular_df[
                            st.session_state.extracurricular_df['student_id'] == selected_student
                        ].copy()
                        
                        if not extracurricular_data.empty:
                            # Convert involvement level to text
                            involvement_map = {
                                1: "Sangat Rendah",
                                2: "Rendah",
                                3: "Sedang",
                                4: "Tinggi",
                                5: "Sangat Tinggi"
                            }
                            
                            extracurricular_data['Tingkat Keterlibatan'] = extracurricular_data['tingkat_keterlibatan'].map(
                                lambda x: involvement_map.get(x, x)
                            )
                            
                            # Select and rename columns
                            extracurricular_display = extracurricular_data[['jenis_ekstrakurikuler', 'kategori', 'Tingkat Keterlibatan', 'prestasi']]
                            extracurricular_display.columns = ['Ekstrakurikuler', 'Kategori', 'Tingkat Keterlibatan', 'Prestasi']
                            
                            # Display table
                            st.dataframe(extracurricular_display, use_container_width=True)
                        else:
                            st.info("Tidak ada data ekstrakurikuler untuk siswa ini.")
                
                # Recommendations tab
                with profile_tabs[3]:
                    st.markdown("<h4>Rekomendasi Jurusan</h4>", unsafe_allow_html=True)
                    
                    # Display recommendation chart
                    rec_chart = plot_recommendation_chart(st.session_state.recommendations, selected_student)
                    st.plotly_chart(rec_chart, use_container_width=True)
                    
                    # Show recommendation details
                    st.markdown("<h4>Detail Rekomendasi</h4>", unsafe_allow_html=True)
                    
                    # Get top 3 recommendations
                    top_jurusan = [(jurusan, student_recs[jurusan]) for jurusan in JURUSAN_CATEGORIES]
                    top_jurusan.sort(key=lambda x: x[1], reverse=True)
                    top_jurusan = top_jurusan[:3]
                    
                    # Display recommendations
                    jurusan_desc = {
                        'STEM': "Jurusan di bidang Sains, Teknologi, Engineering, dan Matematika.",
                        'Kedokteran & Kesehatan': "Jurusan di bidang kedokteran, keperawatan, farmasi, dan kesehatan masyarakat.",
                        'Bisnis & Ekonomi': "Jurusan di bidang manajemen, akuntansi, ekonomi, dan keuangan.",
                        'Ilmu Sosial & Humaniora': "Jurusan di bidang sosial, politik, komunikasi, dan humaniora.",
                        'Seni & Desain': "Jurusan di bidang seni rupa, desain, musik, film, dan arsitektur.",
                        'Pendidikan': "Jurusan di bidang keguruan dan ilmu pendidikan."
                    }
                    
                    for i, (jurusan, score) in enumerate(top_jurusan):
                        st.markdown(f"""
                        <div class='card' style='background-color: {"#ECFDF5" if i == 0 else "#F3F4F6"};'>
                            <h5 style='font-size: 1.1rem; color: #1F2937;'>{i+1}. {jurusan} <span style='float: right;'>Skor: {score:.1f}</span></h5>
                            <p>{jurusan_desc.get(jurusan, '')}</p>
                            <p><strong>Contoh Program Studi:</strong> {get_program_examples(jurusan)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Generate and download report
                    st.markdown("<h4>Laporan Pemetaan Potensi</h4>", unsafe_allow_html=True)
                    
                    if st.button("Generate Laporan Siswa"):
                        # Generate report
                        report = generate_student_report(selected_student, analysis_df, st.session_state.recommendations)
                        report_html = create_report_pdf(report, analysis_df, st.session_state.recommendations)
                        
                        # Create download link
                        b64 = base64.b64encode(report_html.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="Laporan_{selected_student}_{report["nama"]}.html">Download Laporan HTML</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        st.success("Laporan berhasil dibuat. Klik link di atas untuk mengunduh.")
    
    # ===== GROUP ANALYSIS TAB =====
    with tabs[3]:
        st.markdown("<h2 class='sub-header'>Analisis Kelompok</h2>", unsafe_allow_html=True)
        
        if not st.session_state.data_uploaded:
            st.warning("⚠️ Belum ada data yang diunggah. Silakan upload data di tab Manajemen Data.")
        else:
            analysis_df, X_scaled, numerical_cols = preprocess_data()
            
            if analysis_df is not None and st.session_state.cluster_results is not None:
                clusters = st.session_state.cluster_results
                linkage_matrix = st.session_state.linkage_matrix
                
                # Show dendrogram
                st.markdown("<h3 class='section-header'>Dendrogram Hierarchical Clustering</h3>", unsafe_allow_html=True)
                
                dendrogram_fig = plot_dendrogram(linkage_matrix)
                st.pyplot(dendrogram_fig)
                
                # Show cluster distribution
                st.markdown("<h3 class='section-header'>Distribusi Cluster</h3>", unsafe_allow_html=True)
                
                cluster_dist_fig = plot_cluster_distribution(analysis_df, clusters)
                st.plotly_chart(cluster_dist_fig, use_container_width=True)
                
                # Show cluster characteristics
                st.markdown("<h3 class='section-header'>Karakteristik Cluster</h3>", unsafe_allow_html=True)
                
                # Display cluster characteristics heatmap
                cluster_char_fig = plot_cluster_characteristics(analysis_df, clusters, numerical_cols)
                st.plotly_chart(cluster_char_fig, use_container_width=True)
                
                # Show scatter plot of clusters
                st.markdown("<h3 class='section-header'>Visualisasi Cluster 2D</h3>", unsafe_allow_html=True)
                
                # Select features for visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    x_feature = st.selectbox(
                        "Pilih variabel X:",
                        options=numerical_cols,
                        index=0
                    )
                
                with col2:
                    y_feature = st.selectbox(
                        "Pilih variabel Y:",
                        options=numerical_cols,
                        index=min(1, len(numerical_cols)-1)
                    )
                
                # Create scatter plot
                scatter_fig = plot_scatter_clusters(analysis_df, clusters, x_feature, y_feature)
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Show cluster details
                st.markdown("<h3 class='section-header'>Detail Cluster</h3>", unsafe_allow_html=True)
                
                # Create tabs for each cluster
                cluster_tabs = st.tabs([CLUSTER_DESCRIPTIONS[i]['name'] for i in range(len(CLUSTER_DESCRIPTIONS)) if i in np.unique(clusters)])
                
                for i, tab in enumerate([i for i in range(len(CLUSTER_DESCRIPTIONS)) if i in np.unique(clusters)]):
                    with cluster_tabs[i]:
                        cluster_info = CLUSTER_DESCRIPTIONS[tab]
                        
                        # Show cluster description
                        st.markdown(f"<p>{cluster_info['description']}</p>", unsafe_allow_html=True)
                        
                        # Show recommended majors
                        st.markdown("<h4>Rekomendasi Jurusan yang Sesuai</h4>", unsafe_allow_html=True)
                        
                        for jurusan in cluster_info['recommendations']:
                            st.markdown(f"- {jurusan}")
                        
                        # Show students in this cluster
                        st.markdown("<h4>Siswa dalam Cluster Ini</h4>", unsafe_allow_html=True)
                        
                        cluster_students = analysis_df[analysis_df['cluster'] == tab][['student_id', 'nama', 'kelas']].copy()
                        
                        if not cluster_students.empty:
                            st.dataframe(cluster_students, use_container_width=True)
                            
                            # Show student count by class
                            st.markdown("<h4>Distribusi Kelas</h4>", unsafe_allow_html=True)
                            
                            class_counts = cluster_students['kelas'].value_counts().reset_index()
                            class_counts.columns = ['Kelas', 'Jumlah Siswa']
                            
                            # Create bar chart
                            class_fig = px.bar(
                                class_counts,
                                x='Kelas',
                                y='Jumlah Siswa',
                                title="Jumlah Siswa per Kelas",
                                text='Jumlah Siswa'
                            )
                            
                            st.plotly_chart(class_fig, use_container_width=True)
    
    # ===== RECOMMENDATION SYSTEM TAB =====
    with tabs[4]:
        st.markdown("<h2 class='sub-header'>Sistem Rekomendasi</h2>", unsafe_allow_html=True)
        
        if not st.session_state.data_uploaded:
            st.warning("⚠️ Belum ada data yang diunggah. Silakan upload data di tab Manajemen Data.")
        else:
            analysis_df, X_scaled, numerical_cols = preprocess_data()
            
            if analysis_df is not None and st.session_state.recommendations is not None:
                # Explain the recommendation system
                st.markdown("<h3 class='section-header'>Cara Kerja Sistem Rekomendasi</h3>", unsafe_allow_html=True)
                
                st.markdown("""
                Sistem rekomendasi EduMatch mengintegrasikan berbagai aspek kompetensi siswa untuk menghasilkan
                rekomendasi jurusan yang sesuai. Berikut adalah proses yang digunakan:
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    #### 1. Pembobotan Komponen
                    
                    Setiap komponen profil siswa diberi bobot sebagai berikut:
                    - Nilai Akademik: 40%
                    - Soft Skills: 30%
                    - Minat & Bakat: 20%
                    - Ekstrakurikuler: 10%
                    
                    #### 2. Clustering Siswa
                    
                    Siswa dikelompokkan menggunakan metode Hierarchical Clustering berdasarkan kesamaan profil kompetensi.
                    Pendekatan ini membantu mengidentifikasi pola dan kecenderungan dalam data.
                    """)
                
                with col2:
                    st.markdown("""
                    #### 3. Matriks Kesesuaian
                    
                    Setiap faktor profil siswa dihubungkan dengan relevansinya terhadap berbagai jurusan menggunakan
                    matriks kesesuaian. Misalnya, nilai matematika tinggi memiliki relevansi tinggi dengan jurusan STEM.
                    
                    #### 4. Kalkulasi Skor
                    
                    Skor kesesuaian untuk setiap jurusan dihitung berdasarkan formula yang mengintegrasikan
                    berbagai faktor sesuai dengan bobotnya dalam matriks kesesuaian.
                    """)
                
                # Show recommendation matrix
                st.markdown("<h3 class='section-header'>Matriks Kesesuaian</h3>", unsafe_allow_html=True)
                
                # Create a dataframe of the scoring matrix
                scoring_matrix_rows = []
                
                for jurusan, factors in SCORING_MATRIX.items():
                    for factor, weight in factors.items():
                        scoring_matrix_rows.append({
                            'Jurusan': jurusan,
                            'Faktor': factor,
                            'Bobot': weight
                        })
                
                scoring_matrix_df = pd.DataFrame(scoring_matrix_rows)
                
                # Create pivot table
                matrix_pivot = scoring_matrix_df.pivot(index='Faktor', columns='Jurusan', values='Bobot')
                matrix_pivot.fillna(0, inplace=True)
                
                # Create heatmap
                matrix_fig = px.imshow(
                    matrix_pivot,
                    text_auto='.2f',
                    color_continuous_scale='Viridis',
                    title="Matriks Kesesuaian - Bobot Faktor per Jurusan"
                )
                
                st.plotly_chart(matrix_fig, use_container_width=True)
                
                # Show top recommendations
                st.markdown("<h3 class='section-header'>Top Rekomendasi per Siswa</h3>", unsafe_allow_html=True)
                
                # Create plot of top recommendations
                rec_chart = plot_recommendation_chart(st.session_state.recommendations)
                st.plotly_chart(rec_chart, use_container_width=True)
                
                # Show recommendation details
                st.markdown("<h3 class='section-header'>Detail Rekomendasi Jurusan</h3>", unsafe_allow_html=True)
                
                # Select student
                student_options = analysis_df[['student_id', 'nama']].copy()
                student_options['display'] = student_options['student_id'] + ' - ' + student_options['nama']
                
                selected_recommendation = st.selectbox(
                    "Pilih Siswa untuk Detail Rekomendasi:",
                    options=student_options['student_id'].tolist(),
                    format_func=lambda x: student_options[student_options['student_id'] == x]['display'].iloc[0],
                    key="rec_student_select"
                )
                
                # Show detailed calculation for selected student
                if selected_recommendation:
                    student_data = analysis_df[analysis_df['student_id'] == selected_recommendation].iloc[0]
                    student_recs = st.session_state.recommendations[st.session_state.recommendations['student_id'] == selected_recommendation].iloc[0]
                    
                    st.markdown(f"<h4>Detail Perhitungan untuk {student_data['nama']}</h4>", unsafe_allow_html=True)
                    
                    # Create tabs for each jurusan
                    jurusan_tabs = st.tabs(JURUSAN_CATEGORIES)
                    
                    for i, jurusan in enumerate(JURUSAN_CATEGORIES):
                        with jurusan_tabs[i]:
                            # Show calculation details
                            st.markdown(f"<h5>Perhitungan Skor untuk {jurusan}</h5>", unsafe_allow_html=True)
                            
                            # Get factors for this jurusan
                            jurusan_factors = SCORING_MATRIX[jurusan]
                            
                            # Create calculation dataframe
                            calc_rows = []
                            total_score = 0
                            
                            for factor, weight in jurusan_factors.items():
                                if factor in student_data:
                                    value = student_data[factor]
                                    contribution = value * weight
                                    total_score += contribution
                                    
                                    calc_rows.append({
                                        'Faktor': factor,
                                        'Nilai Siswa': value,
                                        'Bobot': weight,
                                        'Kontribusi': contribution
                                    })
                            
                            calc_df = pd.DataFrame(calc_rows)
                            
                            # Add total row
                            calc_df = calc_df.append({
                                'Faktor': 'TOTAL',
                                'Nilai Siswa': '',
                                'Bobot': '',
                                'Kontribusi': total_score
                            }, ignore_index=True)
                            
                            # Show table
                            st.dataframe(calc_df, use_container_width=True)
                            
                            # Show as waterfall chart
                            if len(calc_rows) > 0:
                                # Create a waterfall chart to visualize contributions
                                waterfall_data = calc_df[:-1].copy()  # Exclude total row
                                
                                # Sort by contribution
                                waterfall_data = waterfall_data.sort_values('Kontribusi', ascending=False)
                                
                                fig = go.Figure(go.Waterfall(
                                    name="Skor",
                                    orientation="v",
                                    measure=["relative"] * len(waterfall_data),
                                    x=waterfall_data['Faktor'],
                                    textposition="outside",
                                    text=waterfall_data['Kontribusi'].round(1),
                                    y=waterfall_data['Kontribusi'],
                                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                                ))
                                
                                fig.update_layout(
                                    title=f"Kontribusi Masing-masing Faktor untuk Skor {jurusan}",
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
    
    # ===== DATA MANAGEMENT TAB =====
    with tabs[5]:
        st.markdown("<h2 class='sub-header'>Manajemen Data</h2>", unsafe_allow_html=True)
        
        # Provide detailed instructions
        st.markdown("<h3 class='section-header'>Panduan Pengisian Data</h3>", unsafe_allow_html=True)
        
        with st.expander("Klik untuk Melihat Panduan Pengisian", expanded=True):
            st.markdown("""
            ### Format Data
            
            Untuk menggunakan EduMatch, Anda perlu menyiapkan data dalam format tertentu. Data dapat disiapkan menggunakan template Excel yang disediakan, dengan detail pengisian sebagai berikut:
            
            #### 1️⃣ Data Siswa (Sheet 'students')
            - **student_id**: ID unik untuk setiap siswa (misalnya: S001, S002, dst)
            - **nama**: Nama lengkap siswa
            - **kelas**: Kelas dan jurusan (misalnya: 11 IPA 1, 12 IPS 2)
            - **jenis_kelamin**: L untuk laki-laki, P untuk perempuan
            - **tanggal_lahir**: Format YYYY-MM-DD (misalnya: 2006-07-15)
            
            #### 2️⃣ Nilai Akademik (Sheet 'academic')
            - **student_id**: ID yang sama dengan sheet 'students'
            - Kolom-kolom mata pelajaran seperti **Matematika**, **Fisika**, **Kimia**, dll.
            - Nilai dalam skala 1-100 (misalnya: 85, 92, 78)
            
            #### 3️⃣ Soft Skills (Sheet 'softskills')
            - **student_id**: ID yang sama dengan sheet 'students'
            - Kolom-kolom soft skill: **Komunikasi**, **Kepemimpinan**, **Pemecahan Masalah**, **Kreativitas**, **Kerja Sama Tim**, **Adaptabilitas**, **Manajemen Waktu**, **Kemandirian**
            - Nilai dalam skala 1-100 (bukan 1-5), di mana:
                - 1-20: Sangat kurang
                - 21-40: Kurang
                - 41-60: Cukup
                - 61-80: Baik
                - 81-100: Sangat baik
            
            #### 4️⃣ Minat & Bakat (Sheet 'interests')
            - **student_id**: ID yang sama dengan sheet 'students'
            - Kolom-kolom Holland Code: **Realistic**, **Investigative**, **Artistic**, **Social**, **Enterprising**, **Conventional**
            - Nilai dalam skala 1-100 (bukan 1-5), di mana:
                - 1-20: Sangat tidak sesuai
                - 21-40: Tidak sesuai
                - 41-60: Netral
                - 61-80: Sesuai
                - 81-100: Sangat sesuai
            
            #### 5️⃣ Ekstrakurikuler (Sheet 'extracurricular')
            - **student_id**: ID yang sama dengan sheet 'students' (satu siswa dapat memiliki beberapa baris untuk beberapa ekstrakurikuler)
            - **jenis_ekstrakurikuler**: Nama kegiatan ekstrakurikuler (misalnya: Robotik, OSIS, Basket)
            - **kategori**: Harus dipilih dari opsi berikut:
                - Sains & Teknologi (robotik, komputer, sains, matematika, dll)
                - Sosial & Organisasi (OSIS, PMR, Pramuka, dll)
                - Olahraga (basket, sepak bola, voli, dll)
                - Seni & Budaya (tari, musik, teater, dll)
                - Bahasa (English club, Jepang, debat, dll)
                - Keagamaan (Rohis, Rokris, dll)
            - **tingkat_keterlibatan**: Nilai dalam skala 1-100 yang menunjukkan intensitas partisipasi
            - **prestasi**: "Ya" jika pernah mendapatkan prestasi, "Tidak" jika belum
            
            ### Cara Pengisian Template
            
            1. Unduh template Excel yang disediakan
            2. Isi setiap sheet sesuai format di atas
            3. Pastikan semua siswa memiliki data di semua sheet (kecuali extracurricular yang opsional)
            4. Simpan file Excel tersebut
            5. Unggah file Excel melalui form unggah di bawah
            
            ### Tips Pengisian
            
            - Pastikan ID siswa konsisten di semua sheet
            - Gunakan skala 1-100 untuk semua penilaian (soft skills, minat, tingkat keterlibatan)
            - Untuk ekstrakurikuler, gunakan kategori sesuai dengan daftar yang disediakan
            - Satu siswa dapat memiliki lebih dari satu ekstrakurikuler
            """)
        
        # Data Import Method Selector
        st.markdown("<h3 class='section-header'>Metode Input Data</h3>", unsafe_allow_html=True)
        
        import_method = st.radio(
            "Pilih metode input data:",
            ["Unggah File Excel Lengkap (Semua Sheet)", "Unggah File Terpisah"]
        )
        
        # Create template dataframes - PINDAHKAN KE SINI AGAR TERSEDIA UNTUK SEMUA OPSI IMPORT
        templates = create_template_dataframes()
        
        st.markdown("<h3 class='section-header'>Unduh Template</h3>", unsafe_allow_html=True)
        
        if import_method == "Unggah File Excel Lengkap (Semua Sheet)":
            st.markdown("""
            Untuk memulai penggunaan EduMatch, unduh template Excel berikut dan isi dengan data siswa.
            Template ini berisi semua sheet yang diperlukan dalam satu file Excel.
            """)
            
            # Create multi-sheet Excel for download
            try:
                multi_sheet_href = create_multi_sheet_excel(templates)
                st.markdown(multi_sheet_href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat template Excel: {str(e)}")
                st.markdown("""
                Untuk mengunduh template Excel, Anda memerlukan package xlsxwriter. 
                Silakan install dengan perintah: `pip install xlsxwriter`
                
                Atau Anda bisa mengunduh template per sheet secara terpisah:
                """)
        
        # Always show individual template downloads as backup
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h5>Template Data Siswa</h5>", unsafe_allow_html=True)
            href = download_excel(templates['students'], "EduMatch_Students_Template.xlsx")
            st.markdown(href, unsafe_allow_html=True)
            
            st.markdown("<h5>Template Nilai Akademik</h5>", unsafe_allow_html=True)
            href = download_excel(templates['academic'], "EduMatch_Academic_Template.xlsx")
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h5>Template Soft Skills</h5>", unsafe_allow_html=True)
            href = download_excel(templates['softskills'], "EduMatch_SoftSkills_Template.xlsx")
            st.markdown(href, unsafe_allow_html=True)
            
            st.markdown("<h5>Template Minat & Bakat</h5>", unsafe_allow_html=True)
            href = download_excel(templates['interests'], "EduMatch_Interests_Template.xlsx")
            st.markdown(href, unsafe_allow_html=True)
        
        with col3:
            st.markdown("<h5>Template Ekstrakurikuler</h5>", unsafe_allow_html=True)
            href = download_excel(templates['extracurricular'], "EduMatch_Extracurricular_Template.xlsx")
            st.markdown(href, unsafe_allow_html=True)
            
            # Add help for extracurricular categories
            with st.expander("Kategori Ekstrakurikuler"):
                st.markdown("""
                Berikut adalah daftar kategori ekstrakurikuler yang dapat digunakan:
                
                1. **Sains & Teknologi**
                - Robotik, Komputer, Astronomi, Matematika, Sains, dll.
                
                2. **Sosial & Organisasi**
                - OSIS, MPK, PMR, Pramuka, Jurnalistik, dll.
                
                3. **Olahraga**
                - Basket, Sepak Bola, Voli, Renang, Atletik, dll.
                
                4. **Seni & Budaya**
                - Tari, Musik, Teater, Paduan Suara, Lukis, dll.
                
                5. **Bahasa**
                - English Club, Jepang, Mandarin, Debat, dll.
                
                6. **Keagamaan**
                - Rohis, Rokris, Rokat, dll.
                """)
        
        st.markdown("<h3 class='section-header'>Unggah Data</h3>", unsafe_allow_html=True)
        
        if import_method == "Unggah File Excel Lengkap (Semua Sheet)":
            st.markdown("""
            Unggah file Excel yang berisi semua sheet data siswa. File harus berisi minimal sheet untuk data
            siswa, nilai akademik, soft skills, dan minat & bakat. Pastikan format sesuai dengan template.
            """)
            
            uploaded_file = st.file_uploader("Unggah File Excel Lengkap", type=["xlsx", "xls"], key="full_excel_upload")
            
            if uploaded_file is not None:
                try:
                    # Read Excel file
                    xl = pd.ExcelFile(uploaded_file)
                    
                    # Check if required sheets exist
                    required_sheets = ['students', 'academic', 'softskills', 'interests']
                    missing_sheets = [sheet for sheet in required_sheets if sheet not in xl.sheet_names]
                    
                    if missing_sheets:
                        st.error(f"File Excel tidak valid. Sheet yang diperlukan tidak ditemukan: {', '.join(missing_sheets)}")
                    else:
                        # Read sheets
                        students_df = xl.parse("students")
                        academic_df = xl.parse("academic")
                        softskills_df = xl.parse("softskills")
                        interests_df = xl.parse("interests")
                        
                        # Read extracurricular sheet if it exists
                        if "extracurricular" in xl.sheet_names:
                            extracurricular_df = xl.parse("extracurricular")
                        else:
                            extracurricular_df = pd.DataFrame(columns=['student_id', 'jenis_ekstrakurikuler', 'kategori', 'tingkat_keterlibatan', 'prestasi'])
                        
                        # Validate data
                        validation_errors = []
                        
                        # Check if student_id exists in all dataframes
                        if 'student_id' not in students_df.columns:
                            validation_errors.append("Kolom 'student_id' tidak ditemukan di sheet 'students'")
                        
                        if 'student_id' not in academic_df.columns:
                            validation_errors.append("Kolom 'student_id' tidak ditemukan di sheet 'academic'")
                        
                        if 'student_id' not in softskills_df.columns:
                            validation_errors.append("Kolom 'student_id' tidak ditemukan di sheet 'softskills'")
                        
                        if 'student_id' not in interests_df.columns:
                            validation_errors.append("Kolom 'student_id' tidak ditemukan di sheet 'interests'")
                        
                        # Check if all students have academic, softskills, and interests data
                        if not validation_errors:
                            missing_academic = set(students_df['student_id']) - set(academic_df['student_id'])
                            if missing_academic:
                                validation_errors.append(f"{len(missing_academic)} siswa tidak memiliki data akademik")
                            
                            missing_softskills = set(students_df['student_id']) - set(softskills_df['student_id'])
                            if missing_softskills:
                                validation_errors.append(f"{len(missing_softskills)} siswa tidak memiliki data soft skills")
                            
                            missing_interests = set(students_df['student_id']) - set(interests_df['student_id'])
                            if missing_interests:
                                validation_errors.append(f"{len(missing_interests)} siswa tidak memiliki data minat & bakat")
                        
                        # Display validation errors if any
                        if validation_errors:
                            for error in validation_errors:
                                st.error(error)
                        else:
                            # Store data in session state
                            st.session_state.students_df = students_df
                            st.session_state.academic_df = academic_df
                            st.session_state.softskills_df = softskills_df
                            st.session_state.interests_df = interests_df
                            st.session_state.extracurricular_df = extracurricular_df
                            
                            # Set data_uploaded flag
                            st.session_state.data_uploaded = True
                            
                            # Preprocess data
                            analysis_df, X_scaled, numerical_cols = preprocess_data()
                            
                            if analysis_df is not None and X_scaled is not None:
                                # Perform clustering
                                n_clusters = 4  # Default number of clusters
                                clusters, linkage_matrix = perform_clustering(X_scaled, n_clusters)
                                
                                # Store clustering results
                                st.session_state.cluster_results = clusters
                                st.session_state.linkage_matrix = linkage_matrix
                                
                                # Generate recommendations
                                recommendations = generate_recommendations(analysis_df, clusters)
                                
                                # Store recommendations
                                st.session_state.recommendations = recommendations
                                
                                st.success("Data berhasil diunggah dan dianalisis!")
                            else:
                                st.error("Terjadi kesalahan dalam preprocessing data.")
                
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")
        
        else:  # Upload separate files
            st.markdown("""
            Unggah file Excel terpisah untuk setiap jenis data. Pastikan format sesuai dengan template.
            """)
            
            # Student data upload
            st.subheader("1. Data Siswa")
            students_file = st.file_uploader("Unggah File Data Siswa", type=["xlsx", "xls", "csv"], key="students_upload")
            
            # Academic data upload
            st.subheader("2. Nilai Akademik")
            academic_file = st.file_uploader("Unggah File Nilai Akademik", type=["xlsx", "xls", "csv"], key="academic_upload")
            
            # Soft skills data upload
            st.subheader("3. Soft Skills")
            softskills_file = st.file_uploader("Unggah File Soft Skills", type=["xlsx", "xls", "csv"], key="softskills_upload")
            
            # Interests data upload
            st.subheader("4. Minat & Bakat")
            interests_file = st.file_uploader("Unggah File Minat & Bakat", type=["xlsx", "xls", "csv"], key="interests_upload")
            
            # Extracurricular data upload
            st.subheader("5. Ekstrakurikuler (Opsional)")
            extracurricular_file = st.file_uploader("Unggah File Ekstrakurikuler", type=["xlsx", "xls", "csv"], key="extracurricular_upload")
            
            # Process separate files
            if st.button("Proses Data Terpisah"):
                if students_file is None or academic_file is None or softskills_file is None or interests_file is None:
                    st.error("File data siswa, nilai akademik, soft skills, dan minat & bakat wajib diunggah.")
                else:
                    try:
                        # Read files
                        if students_file.name.endswith('.csv'):
                            students_df = pd.read_csv(students_file)
                        else:
                            students_df = pd.read_excel(students_file)
                            
                        if academic_file.name.endswith('.csv'):
                            academic_df = pd.read_csv(academic_file)
                        else:
                            academic_df = pd.read_excel(academic_file)
                            
                        if softskills_file.name.endswith('.csv'):
                            softskills_df = pd.read_csv(softskills_file)
                        else:
                            softskills_df = pd.read_excel(softskills_file)
                            
                        if interests_file.name.endswith('.csv'):
                            interests_df = pd.read_csv(interests_file)
                        else:
                            interests_df = pd.read_excel(interests_file)
                        
                        # Read extracurricular file if uploaded
                        if extracurricular_file is not None:
                            if extracurricular_file.name.endswith('.csv'):
                                extracurricular_df = pd.read_csv(extracurricular_file)
                            else:
                                extracurricular_df = pd.read_excel(extracurricular_file)
                        else:
                            extracurricular_df = pd.DataFrame(columns=['student_id', 'jenis_ekstrakurikuler', 'kategori', 'tingkat_keterlibatan', 'prestasi'])
                        
                        # Validate data
                        validation_errors = []
                        
                        # Check if student_id exists in all dataframes
                        if 'student_id' not in students_df.columns:
                            validation_errors.append("Kolom 'student_id' tidak ditemukan di data siswa")
                        
                        if 'student_id' not in academic_df.columns:
                            validation_errors.append("Kolom 'student_id' tidak ditemukan di data nilai akademik")
                        
                        if 'student_id' not in softskills_df.columns:
                            validation_errors.append("Kolom 'student_id' tidak ditemukan di data soft skills")
                        
                        if 'student_id' not in interests_df.columns:
                            validation_errors.append("Kolom 'student_id' tidak ditemukan di data minat & bakat")
                        
                        # Check if all students have academic, softskills, and interests data
                        if not validation_errors:
                            missing_academic = set(students_df['student_id']) - set(academic_df['student_id'])
                            if missing_academic:
                                validation_errors.append(f"{len(missing_academic)} siswa tidak memiliki data akademik")
                            
                            missing_softskills = set(students_df['student_id']) - set(softskills_df['student_id'])
                            if missing_softskills:
                                validation_errors.append(f"{len(missing_softskills)} siswa tidak memiliki data soft skills")
                            
                            missing_interests = set(students_df['student_id']) - set(interests_df['student_id'])
                            if missing_interests:
                                validation_errors.append(f"{len(missing_interests)} siswa tidak memiliki data minat & bakat")
                        
                        # Display validation errors if any
                        if validation_errors:
                            for error in validation_errors:
                                st.error(error)
                        else:
                            # Store data in session state
                            st.session_state.students_df = students_df
                            st.session_state.academic_df = academic_df
                            st.session_state.softskills_df = softskills_df
                            st.session_state.interests_df = interests_df
                            st.session_state.extracurricular_df = extracurricular_df
                            
                            # Set data_uploaded flag
                            st.session_state.data_uploaded = True
                            
                            # Preprocess data
                            analysis_df, X_scaled, numerical_cols = preprocess_data()
                            
                            if analysis_df is not None and X_scaled is not None:
                                # Perform clustering
                                n_clusters = 4  # Default number of clusters
                                clusters, linkage_matrix = perform_clustering(X_scaled, n_clusters)
                                
                                # Store clustering results
                                st.session_state.cluster_results = clusters
                                st.session_state.linkage_matrix = linkage_matrix
                                
                                # Generate recommendations
                                recommendations = generate_recommendations(analysis_df, clusters)
                                
                                # Store recommendations
                                st.session_state.recommendations = recommendations
                                
                                st.success("Data berhasil diunggah dan dianalisis!")
                            else:
                                st.error("Terjadi kesalahan dalam preprocessing data.")
                    
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")
        
        # Show data preview if uploaded
        if st.session_state.data_uploaded:
            st.markdown("<h3 class='section-header'>Preview Data</h3>", unsafe_allow_html=True)
            
            data_tabs = st.tabs(["Siswa", "Akademik", "Soft Skills", "Minat & Bakat", "Ekstrakurikuler"])
            
            with data_tabs[0]:
                st.dataframe(st.session_state.students_df, use_container_width=True)
            
            with data_tabs[1]:
                st.dataframe(st.session_state.academic_df, use_container_width=True)
            
            with data_tabs[2]:
                st.dataframe(st.session_state.softskills_df, use_container_width=True)
            
            with data_tabs[3]:
                st.dataframe(st.session_state.interests_df, use_container_width=True)
            
            with data_tabs[4]:
                st.dataframe(st.session_state.extracurricular_df, use_container_width=True)
            
            # Add option to reset data
            if st.button("Reset Data"):
                st.session_state.data_uploaded = False
                st.session_state.students_df = None
                st.session_state.academic_df = None
                st.session_state.softskills_df = None
                st.session_state.interests_df = None
                st.session_state.extracurricular_df = None
                st.session_state.cluster_results = None
                st.session_state.recommendations = None
                st.session_state.selected_student = None
                
                st.success("Data berhasil direset.")
                st.experimental_rerun()

# Helper function to get program examples based on jurusan category
def get_program_examples(jurusan):
    program_examples = {
        'STEM': "Teknik Informatika, Ilmu Komputer, Matematika, Fisika, Teknik Elektro, Teknik Sipil",
        'Kedokteran & Kesehatan': "Pendidikan Dokter, Keperawatan, Farmasi, Gizi, Kesehatan Masyarakat, Kedokteran Gigi",
        'Bisnis & Ekonomi': "Manajemen, Akuntansi, Ekonomi Pembangunan, Bisnis Digital, Kewirausahaan, Perbankan",
        'Ilmu Sosial & Humaniora': "Psikologi, Ilmu Komunikasi, Hubungan Internasional, Sosiologi, Ilmu Politik, Sastra",
        'Seni & Desain': "Desain Komunikasi Visual, Seni Rupa, Arsitektur, Desain Interior, Desain Produk, Film & Animasi",
        'Pendidikan': "Pendidikan Matematika, Pendidikan Bahasa, PGSD, Pendidikan IPA, Teknologi Pendidikan, Bimbingan Konseling"
    }
    
    return program_examples.get(jurusan, "")

# Run the app
if __name__ == "__main__":
    main()