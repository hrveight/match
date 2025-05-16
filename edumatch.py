import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# ===== Styling =====
st.set_page_config(page_title="EduMatch Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.8rem; color: #2563EB; margin-top: 1.5rem; margin-bottom: 1rem;}
    .section-header {font-size: 1.4rem; color: #3B82F6; margin-top: 1rem; margin-bottom: 0.5rem;}
    .card {background-color: #F3F4F6; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;}
    .metric-card {background-color: #EFF6FF; border-radius: 0.5rem; padding: 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.12);}
    .metric-value {font-size: 1.8rem; font-weight: bold; color: #1E40AF;}
    .metric-label {font-size: 0.9rem; color: #4B5563;}
    .info-text {font-size: 0.9rem; color: #4B5563;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #F3F4F6; border-radius: 4px 4px 0 0; padding: 10px 10px;}
    .stTabs [aria-selected="true"] {background-color: #DBEAFE;}
</style>
""", unsafe_allow_html=True)

# ===== Constants =====
SUBJECT_CATEGORIES = {
    'Sains': ['Matematika', 'Fisika', 'Kimia', 'Biologi'],
    'Sosial': ['Sejarah', 'Ekonomi', 'Geografi', 'Sosiologi'],
    'Bahasa': ['B. Indonesia', 'B. Inggris'],
    'Lainnya': ['Seni', 'Olahraga', 'Agama', 'PKN']
}

SOFT_SKILLS = ['Komunikasi', 'Kepemimpinan', 'Pemecahan Masalah', 'Kreativitas', 'Kerja Sama Tim', 'Adaptabilitas', 'Manajemen Waktu', 'Kemandirian']

HOLLAND_INTERESTS = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']

EXTRACURRICULAR_CATEGORIES = ['Sains & Teknologi', 'Sosial & Organisasi', 'Olahraga', 'Seni & Budaya', 'Bahasa', 'Keagamaan']

JURUSAN_CATEGORIES = ['STEM', 'Kedokteran & Kesehatan', 'Bisnis & Ekonomi', 'Ilmu Sosial & Humaniora', 'Seni & Desain', 'Pendidikan']

CLUSTER_DESCRIPTIONS = {
    0: {'name': 'Academically Strong with Technical Skills',
        'description': 'Siswa dengan nilai tinggi di sains, pemecahan masalah, dan aktif di ekskul teknologi.',
        'recommendations': ['STEM', 'Kedokteran & Kesehatan', 'Teknik']},
    1: {'name': 'Creative-Practical Learners',
        'description': 'Siswa kreatif dan adaptif, lebih unggul di seni dan olahraga.',
        'recommendations': ['Seni & Desain', 'Teknik Terapan', 'Kesehatan']},
    2: {'name': 'Socially Competent Communicators',
        'description': 'Siswa dengan kemampuan komunikasi dan sosial tinggi, aktif di organisasi.',
        'recommendations': ['Ilmu Sosial & Humaniora', 'Bisnis & Ekonomi', 'Pendidikan']},
    3: {'name': 'Underperforming with Growth Potential',
        'description': 'Siswa dengan nilai di bawah rata-rata namun berpotensi dengan pendampingan.',
        'recommendations': ['Program Pengembangan Soft Skills', 'Eksplorasi Minat', 'Motivasi Akademik']}
}

SCORING_MATRIX = {
    'STEM': {'Sains': 0.5, 'Sosial': 0.1, 'Bahasa': 0.1, 'Lainnya': 0.05, 'Pemecahan Masalah': 0.15, 'Investigative': 0.25, 'Ekskul Sains & Teknologi': 0.1},
    'Kedokteran & Kesehatan': {'Sains': 0.6, 'Sosial': 0.1, 'Bahasa': 0.1, 'Lainnya': 0.05, 'Kerja Sama Tim': 0.1, 'Social': 0.15, 'Investigative': 0.1, 'Ekskul Sosial & Organisasi': 0.05},
    'Bisnis & Ekonomi': {'Sains': 0.2, 'Sosial': 0.3, 'Bahasa': 0.2, 'Lainnya': 0.05, 'Kepemimpinan': 0.15, 'Enterprising': 0.2, 'Conventional': 0.1, 'Ekskul Sosial & Organisasi': 0.05},
    'Ilmu Sosial & Humaniora': {'Sains': 0.1, 'Sosial': 0.25, 'Bahasa': 0.3, 'Lainnya': 0.05, 'Komunikasi': 0.15, 'Social': 0.2, 'Artistic': 0.05, 'Ekskul Sosial & Organisasi': 0.05},
    'Seni & Desain': {'Sains': 0.1, 'Sosial': 0.1, 'Bahasa': 0.2, 'Lainnya': 0.2, 'Kreativitas': 0.25, 'Artistic': 0.3, 'Ekskul Seni & Budaya': 0.1},
    'Pendidikan': {'Sains': 0.15, 'Sosial': 0.2, 'Bahasa': 0.25, 'Lainnya': 0.1, 'Komunikasi': 0.2, 'Social': 0.25, 'Ekskul Sosial & Organisasi': 0.05}
}

# ===== Session State Initialization =====
for key in ['data_uploaded', 'students_df', 'academic_df', 'softskills_df', 'interests_df', 'extracurricular_df',
            'cluster_results', 'linkage_matrix', 'recommendations', 'students_with_cluster', 'selected_student']:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state['data_uploaded'] is None:
    st.session_state['data_uploaded'] = False

# ===== Utility Functions =====

def create_template_dataframes():
    students = pd.DataFrame({
        'student_id': ['S001', 'S002'],
        'nama': ['John Doe', 'Jane Smith'],
        'kelas': ['11 IPA 1', '11 IPA 2'],
        'jenis_kelamin': ['L', 'P'],
        'tanggal_lahir': ['2006-05-15', '2006-08-21']
    })
    academic = pd.DataFrame({'student_id': ['S001', 'S002']})
    for sublist in SUBJECT_CATEGORIES.values():
        for sub in sublist:
            academic[sub] = [80, 75]
    softskills = pd.DataFrame({'student_id': ['S001', 'S002']})
    for skill in SOFT_SKILLS:
        softskills[skill] = [4, 3]
    interests = pd.DataFrame({'student_id': ['S001', 'S002']})
    for interest in HOLLAND_INTERESTS:
        interests[interest] = [3, 4]
    extracurricular = pd.DataFrame({
        'student_id': ['S001', 'S002'],
        'jenis_ekstrakurikuler': ['Robotik', 'OSIS'],
        'kategori': ['Sains & Teknologi', 'Sosial & Organisasi'],
        'tingkat_keterlibatan': [5, 4],
        'prestasi': ['Ya', 'Tidak']
    })
    return {'students': students, 'academic': academic, 'softskills': softskills, 'interests': interests, 'extracurricular': extracurricular}

def create_multi_sheet_excel(dfs_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    data = output.getvalue()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="EduMatch_Templates.xlsx">Download All Templates (Excel)</a>'
    return href

def download_excel(df, filename):
    towrite = BytesIO()
    df.to_excel(towrite, index=False, engine='xlsxwriter')
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def validate_and_store_data(students_df, academic_df, softskills_df, interests_df, extracurricular_df):
    errors = []
    dfs = {'students': students_df, 'academic': academic_df, 'softskills': softskills_df, 'interests': interests_df}
    for name, df in dfs.items():
        if 'student_id' not in df.columns:
            errors.append(f"Kolom 'student_id' tidak ditemukan di data {name}")
    if errors:
        return errors
    student_ids = set(students_df['student_id'])
    for name, df in dfs.items():
        missing = student_ids - set(df['student_id'])
        if missing:
            errors.append(f"{len(missing)} siswa tidak memiliki data {name} (contoh: {list(missing)[:3]})")
    if errors:
        return errors
    # Save to session_state
    st.session_state.students_df = students_df
    st.session_state.academic_df = academic_df
    st.session_state.softskills_df = softskills_df
    st.session_state.interests_df = interests_df
    st.session_state.extracurricular_df = extracurricular_df
    st.session_state.data_uploaded = True
    return []

def preprocess_data():
    if not st.session_state.data_uploaded:
        return None, None, None
    acad = st.session_state.academic_df.copy()
    academic_cat = {}
    for cat, subs in SUBJECT_CATEGORIES.items():
        subs_valid = [s for s in subs if s in acad.columns]
        if subs_valid:
            academic_cat[cat] = acad[subs_valid].mean(axis=1)
    academic_cat_df = pd.DataFrame(academic_cat)
    academic_cat_df['student_id'] = acad['student_id']
    soft_df = st.session_state.softskills_df.copy()
    for skill in SOFT_SKILLS:
        if skill in soft_df.columns:
            soft_df[skill] = (soft_df[skill] - 1) * 25
    int_df = st.session_state.interests_df.copy()
    for interest in HOLLAND_INTERESTS:
        if interest in int_df.columns:
            int_df[interest] = (int_df[interest] - 1) * 25
    extra_df = st.session_state.extracurricular_df.copy()
    merged = st.session_state.students_df[['student_id']].copy()
    for cat in EXTRACURRICULAR_CATEGORIES:
        cat_df = extra_df[extra_df['kategori'] == cat].copy()
        if not cat_df.empty:
            cat_df['score'] = (cat_df['tingkat_keterlibatan'] - 1) * 25
            cat_df.loc[cat_df['prestasi'] == 'Ya', 'score'] += 20
            cat_df['score'] = cat_df['score'].clip(upper=100)
            cat_score = cat_df.groupby('student_id')['score'].max().reset_index()
            merged = merged.merge(cat_score, on='student_id', how='left')
            merged.rename(columns={'score': f'Ekskul {cat}'}, inplace=True)
    merged.fillna(0, inplace=True)
    df = st.session_state.students_df.merge(academic_cat_df, on='student_id', how='left')
    for skill in SOFT_SKILLS:
        if skill in soft_df.columns:
            df = df.merge(soft_df[['student_id', skill]], on='student_id', how='left')
    for interest in HOLLAND_INTERESTS:
        if interest in int_df.columns:
            df = df.merge(int_df[['student_id', interest]], on='student_id', how='left')
    df = df.merge(merged, on='student_id', how='left', suffixes=('', '_extrakurikuler'))
    df.fillna(0, inplace=True)
    features = list(academic_cat.keys()) + SOFT_SKILLS + HOLLAND_INTERESTS + [f'Ekskul {cat}' for cat in EXTRACURRICULAR_CATEGORIES]
    features = [f for f in features if f in df.columns]
    if not features:
        st.error("Tidak ada fitur numerik untuk analisis.")
        return None, None, None
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, features

def perform_clustering(X, n_clusters=4):
    if X is None:
        return None, None
    linkage_matrix = linkage(X, method='ward')
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = hc.fit_predict(X)
    return clusters, linkage_matrix

def generate_recommendations(df, clusters):
    if df is None or clusters is None:
        return None
    df['cluster'] = clusters
    recs = pd.DataFrame({'student_id': df['student_id']})
    for jurusan in JURUSAN_CATEGORIES:
        score = np.zeros(len(df))
        for factor, weight in SCORING_MATRIX[jurusan].items():
            if factor in df.columns:
                score += df[factor] * weight
        recs[jurusan] = score
    recs['cluster'] = clusters
    recs = recs.merge(df[['student_id', 'nama']], on='student_id', how='left')
    return recs

def preprocess_and_cluster():
    analysis_df, X_scaled, numerical_cols = preprocess_data()
    if analysis_df is None:
        return None, None, None, None
    clusters, linkage_matrix = perform_clustering(X_scaled)
    if clusters is not None:
        analysis_df['cluster'] = clusters
    return analysis_df, X_scaled, numerical_cols, linkage_matrix

# ===== Visualization Functions =====

def plot_dendrogram(linkage_matrix):
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linkage_matrix, ax=ax)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Index Siswa')
    plt.ylabel('Jarak')
    return fig

def plot_cluster_distribution(df, clusters):
    counts = pd.Series(clusters).value_counts().sort_index()
    names = [CLUSTER_DESCRIPTIONS[i]['name'] for i in counts.index]
    fig = px.pie(names=names, values=counts.values, title="Distribusi Cluster Siswa")
    return fig

def plot_radar_chart(student_data, cols, title):
    import plotly.graph_objects as go
    categories = cols.copy()
    values = student_data[cols].values.flatten().tolist()
    values += values[:1]
    categories += categories[:1]
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title=title)
    return fig

def plot_recommendation_chart(recs, student_id=None):
    if student_id:
        student = recs[recs['student_id'] == student_id].iloc[0]
        jurusan_scores = {j: student[j] for j in JURUSAN_CATEGORIES}
        jurusan_df = pd.DataFrame({'Jurusan': list(jurusan_scores.keys()), 'Skor': list(jurusan_scores.values())})
        jurusan_df = jurusan_df.sort_values('Skor', ascending=False)
        fig = px.bar(jurusan_df, x='Jurusan', y='Skor', title=f"Rekomendasi Jurusan untuk {student['nama']}", color='Skor', color_continuous_scale='Viridis')
        return fig
    else:
        top_recs = []
        for _, row in recs.iterrows():
            sorted_jurusan = sorted([(j, row[j]) for j in JURUSAN_CATEGORIES], key=lambda x: x[1], reverse=True)[:2]
            for rank, (jur, score) in enumerate(sorted_jurusan):
                top_recs.append({'Nama': row['nama'], 'Jurusan': jur, 'Skor': score, 'Peringkat': f'Rekomendasi {rank+1}'})
        df_top = pd.DataFrame(top_recs)
        fig = px.bar(df_top, x='Nama', y='Skor', color='Peringkat', barmode='group', title='Top 2 Rekomendasi Jurusan per Siswa')
        return fig

def plot_cluster_characteristics(analysis_df, clusters, numerical_cols):
    cluster_means = pd.DataFrame()
    for i in range(len(CLUSTER_DESCRIPTIONS)):
        if i in np.unique(clusters):
            cluster_data = analysis_df[analysis_df['cluster'] == i]
            if not cluster_data.empty:
                cluster_means[f'Cluster {i}'] = cluster_data[numerical_cols].mean()
    fig = px.imshow(cluster_means,
                    labels=dict(x="Cluster", y="Feature", color="Value"),
                    x=cluster_means.columns,
                    y=cluster_means.index,
                    color_continuous_scale="RdBu_r",
                    title="Karakteristik Cluster (Nilai Rata-rata)",
                    aspect="auto")
    fig.update_layout(height=800)
    return fig

def plot_scatter_clusters(analysis_df, clusters, feature_x, feature_y):
    scatter_df = pd.DataFrame({
        'student_id': analysis_df['student_id'],
        'nama': analysis_df['nama'],
        'x': analysis_df[feature_x],
        'y': analysis_df[feature_y],
        'cluster': [CLUSTER_DESCRIPTIONS[c]['name'] for c in clusters]
    })
    fig = px.scatter(scatter_df,
                     x='x',
                     y='y',
                     color='cluster',
                     hover_name='nama',
                     title=f'Clustering Siswa: {feature_x} vs {feature_y}',
                     labels={'x': feature_x, 'y': feature_y},
                     color_discrete_sequence=px.colors.qualitative.Safe)
    return fig

# ===== Main Function =====

def main():
    st.markdown("<h1 class='main-header'>EduMatch Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.2rem;'>Sistem Pemetaan Potensi dan Rekomendasi Jurusan untuk Siswa SMA/SMK</p>", unsafe_allow_html=True)

    tabs = st.tabs(["🏠 Beranda", "📊 Dashboard Overview", "👤 Analisis Individual", "👥 Analisis Kelompok", "🎯 Sistem Rekomendasi", "🗃️ Manajemen Data"])

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
        
        if st.session_state.data_uploaded:
            st.success("✅ Data telah diunggah dan siap dianalisis. Silakan jelajahi tab-tab lainnya.")
        else:
            st.warning("⚠️ Belum ada data yang diunggah. Silakan unduh template dan upload data di tab Manajemen Data.")
            
            st.markdown("<h3 class='section-header'>Preview Dashboard</h3>", unsafe_allow_html=True)
            st.image("https://via.placeholder.com/800x400?text=EduMatch+Dashboard+Preview", use_container_width=True)

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
            - Nilai dalam skala 1-5, di mana:
                - 1: Sangat kurang
                - 2: Kurang
                - 3: Cukup
                - 4: Baik
                - 5: Sangat baik
            
            #### 4️⃣ Minat & Bakat (Sheet 'interests')
            - **student_id**: ID yang sama dengan sheet 'students'
            - Kolom-kolom Holland Code: **Realistic**, **Investigative**, **Artistic**, **Social**, **Enterprising**, **Conventional**
            - Nilai dalam skala 1-5, di mana:
                - 1: Sangat tidak sesuai
                - 2: Tidak sesuai
                - 3: Netral
                - 4: Sesuai
                - 5: Sangat sesuai
            
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
            - **tingkat_keterlibatan**: Nilai dalam skala 1-5 yang menunjukkan intensitas partisipasi
            - **prestasi**: "Ya" jika pernah mendapatkan prestasi, "Tidak" jika belum
            
            ### Cara Pengisian Template
            
            1. Unduh template Excel yang disediakan
            2. Isi setiap sheet sesuai format di atas
            3. Pastikan semua siswa memiliki data di semua sheet (kecuali extracurricular yang opsional)
            4. Simpan file Excel tersebut
            5. Unggah file Excel melalui form unggah di bawah
            
            ### Tips Pengisian
            
            - Pastikan ID siswa konsisten di semua sheet
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

        # File Upload Section
        st.markdown("<h3 class='section-header'>Unggah Data</h3>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Unggah file Excel (multi-sheet):", type=["xlsx", "xls"])
        if uploaded:
            try:
                xl = pd.ExcelFile(uploaded)
                req = ['students', 'academic', 'softskills', 'interests']
                missing = [sheet for sheet in req if sheet not in xl.sheet_names]
                if missing:
                    st.error(f"Sheet berikut hilang: {', '.join(missing)}")
                else:
                    students_df = xl.parse('students')
                    academic_df = xl.parse('academic')
                    softskills_df = xl.parse('softskills')
                    interests_df = xl.parse('interests')
                    extracurricular_df = xl.parse('extracurricular') if 'extracurricular' in xl.sheet_names else pd.DataFrame(columns=['student_id','jenis_ekstrakurikuler','kategori','tingkat_keterlibatan','prestasi'])

                    errors = validate_and_store_data(students_df, academic_df, softskills_df, interests_df, extracurricular_df)
                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        st.success("Data berhasil diunggah dan valid!")

                        analysis_df, X_scaled, numerical_cols, linkage_matrix = preprocess_and_cluster()
                        if analysis_df is not None:
                            st.session_state.students_with_cluster = analysis_df
                            st.session_state.cluster_results = analysis_df['cluster'].values
                            st.session_state.linkage_matrix = linkage_matrix
                            st.session_state.recommendations = generate_recommendations(analysis_df, analysis_df['cluster'].values)

            except Exception as e:
                st.error(f"Error membaca file: {e}")

    # Tab Dashboard Overview
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Dashboard Overview</h2>", unsafe_allow_html=True)
        if not st.session_state.data_uploaded:
            st.warning("Unggah data dulu di tab Manajemen Data.")
        else:
            df = st.session_state.students_with_cluster
            clusters = st.session_state.cluster_results
            if df is not None and clusters is not None:
                st.markdown(f"Total siswa: {len(df)}")
                st.plotly_chart(plot_cluster_distribution(df, clusters), use_container_width=True, key="cluster_distribution_overview")

    # Tab Analisis Individual
    with tabs[2]:
        st.markdown("<h2 class='sub-header'>Analisis Individual</h2>", unsafe_allow_html=True)
        if not st.session_state.data_uploaded:
            st.warning("⚠️ Belum ada data yang diunggah. Silakan upload data di tab Manajemen Data.")
        else:
            analysis_df = st.session_state.students_with_cluster
            recommendations = st.session_state.recommendations
            if analysis_df is None or recommendations is None:
                st.warning("Data belum siap. Silakan upload ulang data.")
            else:
                student_options = analysis_df[['student_id', 'nama']].copy()
                student_options['display'] = student_options['student_id'] + ' - ' + student_options['nama']

                selected_student = st.selectbox(
                    "Pilih Siswa:",
                    options=student_options['student_id'].tolist(),
                    format_func=lambda x: student_options[student_options['student_id'] == x]['display'].iloc[0],
                    key="select_student_individual"
                )
                st.session_state.selected_student = selected_student

                student_data = analysis_df[analysis_df['student_id'] == selected_student].iloc[0]
                student_recs = recommendations[recommendations['student_id'] == selected_student].iloc[0]

                gender = "Laki-laki" if student_data['jenis_kelamin'] == 'L' else "Perempuan"
                cluster_name = CLUSTER_DESCRIPTIONS[int(student_data['cluster'])]['name']

                st.markdown(f"""
                <div class='card'>
                    <p><strong>Nama:</strong> {student_data['nama']}</p>
                    <p><strong>ID Siswa:</strong> {selected_student}</p>
                    <p><strong>Kelas:</strong> {student_data['kelas']}</p>
                    <p><strong>Jenis Kelamin:</strong> {gender}</p>
                    <p><strong>Tanggal Lahir:</strong> {student_data.get('tanggal_lahir', '-')}</p>
                    <p><strong>Cluster:</strong> {cluster_name}</p>
                </div>
                """, unsafe_allow_html=True)

                profile_tabs = st.tabs(["Akademik", "Soft Skills", "Minat & Bakat", "Rekomendasi Jurusan"])

                # Tab Akademik
                with profile_tabs[0]:
                    st.markdown("<h4>Profil Akademik</h4>", unsafe_allow_html=True)
                    academic_cols = []
                    for cat in SUBJECT_CATEGORIES:
                        academic_cols.extend([s for s in SUBJECT_CATEGORIES[cat] if s in analysis_df.columns])
                    if academic_cols:
                        st.plotly_chart(plot_radar_chart(student_data.to_frame().T, academic_cols, "Profil Akademik - Kategori"), use_container_width=True, key="radar_akademik_individual")

                    st.markdown("<h4>Detail Nilai Akademik</h4>", unsafe_allow_html=True)
                    subject_cols = academic_cols
                    student_subjects = st.session_state.academic_df[st.session_state.academic_df['student_id'] == selected_student]
                    if not student_subjects.empty:
                        subjects_data = []
                        for subject in subject_cols:
                            if subject in student_subjects.columns:
                                category = next((c for c, s in SUBJECT_CATEGORIES.items() if subject in s), "Lainnya")
                                subjects_data.append({'Mata Pelajaran': subject, 'Kategori': category, 'Nilai': student_subjects[subject].iloc[0]})
                        subjects_df = pd.DataFrame(subjects_data)
                        if not subjects_df.empty:
                            subjects_df = subjects_df.sort_values(['Kategori', 'Nilai'], ascending=[True, False])
                            bar_fig = px.bar(subjects_df, x='Mata Pelajaran', y='Nilai', color='Kategori', title="Nilai Detail per Mata Pelajaran", text='Nilai')
                            bar_fig.update_layout(xaxis_title="Mata Pelajaran", yaxis_title="Nilai", yaxis_range=[0,100])
                            st.plotly_chart(bar_fig, use_container_width=True, key="bar_nilai_akademik_individual")

                # Tab Soft Skills
                with profile_tabs[1]:
                    st.markdown("<h4>Profil Soft Skills</h4>", unsafe_allow_html=True)
                    softskill_cols = [col for col in SOFT_SKILLS if col in analysis_df.columns]
                    if softskill_cols:
                        st.plotly_chart(plot_radar_chart(student_data.to_frame().T, softskill_cols, "Profil Soft Skills"), use_container_width=True, key="radar_softskills_individual")
                        st.markdown("<h4>Perbandingan dengan Rata-rata Kelas</h4>", unsafe_allow_html=True)
                        class_avg = analysis_df[analysis_df['kelas'] == student_data['kelas']][softskill_cols].mean()
                        comparison_df = pd.DataFrame({'Soft Skill': softskill_cols, 'Nilai Siswa': [student_data[s] for s in softskill_cols], 'Rata-rata Kelas': [class_avg[s] for s in softskill_cols]})
                        comp_fig = go.Figure()
                        comp_fig.add_trace(go.Bar(x=comparison_df['Soft Skill'], y=comparison_df['Nilai Siswa'], name='Nilai Siswa', marker_color='#3B82F6'))
                        comp_fig.add_trace(go.Bar(x=comparison_df['Soft Skill'], y=comparison_df['Rata-rata Kelas'], name='Rata-rata Kelas', marker_color='#9CA3AF'))
                        comp_fig.update_layout(title="Perbandingan Soft Skills dengan Rata-rata Kelas", xaxis_title="Soft Skill", yaxis_title="Nilai", barmode='group', yaxis_range=[0,100])
                        st.plotly_chart(comp_fig, use_container_width=True, key="compare_softskills_individual")

                # Tab Minat & Bakat
                with profile_tabs[2]:
                    st.markdown("<h4>Profil Minat & Bakat</h4>", unsafe_allow_html=True)
                    interest_cols = [col for col in HOLLAND_INTERESTS if col in analysis_df.columns]
                    if interest_cols:
                        st.plotly_chart(plot_radar_chart(student_data.to_frame().T, interest_cols, "Profil RIASEC (Holland Code)"), use_container_width=True, key="radar_interest_individual")

                # Tab Rekomendasi Jurusan
                with profile_tabs[3]:
                    st.markdown("<h4>Rekomendasi Jurusan</h4>", unsafe_allow_html=True)
                    st.plotly_chart(plot_recommendation_chart(st.session_state.recommendations, selected_student), use_container_width=True, key="recommendation_student_individual")

    # Tab Analisis Kelompok
    with tabs[3]:
        st.markdown("<h2 class='sub-header'>Analisis Kelompok</h2>", unsafe_allow_html=True)
        if not st.session_state.data_uploaded:
            st.warning("⚠️ Belum ada data yang diunggah. Silakan upload data di tab Manajemen Data.")
        else:
            analysis_df = st.session_state.students_with_cluster
            clusters = st.session_state.cluster_results
            linkage_matrix = st.session_state.linkage_matrix
            if analysis_df is not None and clusters is not None and linkage_matrix is not None:
                st.markdown("<h3 class='section-header'>Dendrogram Hierarchical Clustering</h3>", unsafe_allow_html=True)
                st.pyplot(plot_dendrogram(linkage_matrix), clear_figure=True)

                st.markdown("<h3 class='section-header'>Distribusi Cluster</h3>", unsafe_allow_html=True)
                st.plotly_chart(plot_cluster_distribution(analysis_df, clusters), use_container_width=True, key="cluster_distribution_group")

                st.markdown("<h3 class='section-header'>Karakteristik Cluster</h3>", unsafe_allow_html=True)
                numerical_cols = [col for col in analysis_df.columns if col not in ['student_id', 'nama', 'kelas', 'jenis_kelamin', 'tanggal_lahir', 'cluster']]
                st.plotly_chart(plot_cluster_characteristics(analysis_df, clusters, numerical_cols), use_container_width=True, key="cluster_characteristics_group")

                st.markdown("<h3 class='section-header'>Visualisasi Cluster 2D</h3>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("Pilih variabel X:", options=numerical_cols, index=0, key="scatter_x_feature")
                with col2:
                    y_feature = st.selectbox("Pilih variabel Y:", options=numerical_cols, index=min(1,len(numerical_cols)-1), key="scatter_y_feature")
                st.plotly_chart(plot_scatter_clusters(analysis_df, clusters, x_feature, y_feature), use_container_width=True, key="scatter_clusters_group")

    # Tab Sistem Rekomendasi
    with tabs[4]:
        st.markdown("<h2 class='sub-header'>Sistem Rekomendasi</h2>", unsafe_allow_html=True)
        if not st.session_state.data_uploaded:
            st.warning("⚠️ Belum ada data yang diunggah. Silakan upload data di tab Manajemen Data.")
        else:
            analysis_df = st.session_state.students_with_cluster
            recommendations = st.session_state.recommendations
            if analysis_df is not None and recommendations is not None:
                st.markdown("<h3 class='section-header'>Top Rekomendasi per Siswa</h3>", unsafe_allow_html=True)
                st.plotly_chart(plot_recommendation_chart(recommendations), use_container_width=True, key="recommendation_all_students")

                student_options = analysis_df[['student_id','nama']].copy()
                student_options['display'] = student_options['student_id'] + ' - ' + student_options['nama']
                selected_recommendation = st.selectbox(
                    "Pilih Siswa untuk Detail Rekomendasi:",
                    options=student_options['student_id'].tolist(),
                    format_func=lambda x: student_options[student_options['student_id'] == x]['display'].iloc[0],
                    key="rec_student_select"
                )
                if selected_recommendation:
                    student_data = analysis_df[analysis_df['student_id'] == selected_recommendation].iloc[0]
                    student_recs = recommendations[recommendations['student_id'] == selected_recommendation].iloc[0]
                    st.markdown(f"<h4>Detail Perhitungan untuk {student_data['nama']}</h4>", unsafe_allow_html=True)
                    jurusan_tabs = st.tabs(JURUSAN_CATEGORIES)
                    for i, jurusan in enumerate(JURUSAN_CATEGORIES):
                        with jurusan_tabs[i]:
                            st.markdown(f"<h5>Perhitungan Skor untuk {jurusan}</h5>", unsafe_allow_html=True)
                            jurusan_factors = SCORING_MATRIX[jurusan]
                            calc_rows = []
                            total_score = 0
                            for factor, weight in jurusan_factors.items():
                                if factor in student_data:
                                    value = student_data[factor]
                                    contribution = value * weight
                                    total_score += contribution
                                    calc_rows.append({'Faktor': factor, 'Nilai Siswa': value, 'Bobot': weight, 'Kontribusi': contribution})
                            calc_df = pd.DataFrame(calc_rows)
                            total_row = pd.DataFrame([{'Faktor': 'TOTAL', 'Nilai Siswa': '', 'Bobot': '', 'Kontribusi': total_score}])
                            calc_df = pd.concat([calc_df, total_row], ignore_index=True)
                            st.dataframe(calc_df, use_container_width=True)

if __name__ == "__main__":
    main()
