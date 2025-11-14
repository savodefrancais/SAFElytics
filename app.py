"""
✨ SAFElytics - ZERO CODE. FULL POWER. ✨
================================================
A comprehensive Streamlit web application for data processing and machine learning without coding.
Developed by SAFE AI-AFRICA

KEY FEATURES:
✅ Multi-format data upload (CSV, Excel, JSON)
✅ Interactive data preview with sorting & filtering
✅ Comprehensive data preprocessing:
   - Handle missing values (7+ strategies)
   - Categorical encoding (Label & One-Hot)
   - Feature scaling (Standardization & Normalization)
   - Feature selection & duplicate removal
✅ Advanced visualizations:
   - Bar charts, Line charts, Scatter plots
   - Histograms, Correlation heatmaps
✅ No-code ML model building:
   Classification: Logistic Regression, Random Forest, SVM, KNN
   Regression: Linear Regression, Random Forest Regressor
   Clustering: KMeans
✅ Comprehensive model evaluation
✅ Download processed data & trained models
✅ Clean, responsive UI with full error handling

INSTALLATION:
pip install streamlit pandas numpy matplotlib seaborn scikit-learn openpyxl joblib

USAGE:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score
)
import joblib
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="✨ SAFElytics - Zero Code. Full Power.",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 1rem 2rem; }
    h1 { color: #1f77b4; }
    h2 { color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }
    
    /* Fix metric visibility */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    
    [data-testid="metric-container"] > div > div {
        color: #0d5aa7 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="metric-container"] > div:last-child > div {
        color: #1f77b4 !important;
        font-weight: 700 !important;
        font-size: 24px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = {}
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None

init_session_state()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_dataset(uploaded_file):
    """Load dataset from uploaded file (CSV, Excel, JSON, or ZIP)"""
    try:
        filename = uploaded_file.name.lower()
        
        if filename.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        elif filename.endswith('.json'):
            return pd.read_json(uploaded_file)
        elif filename.endswith('.zip'):
            import zipfile
            # Extract ZIP and find data files
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                # Get list of files in ZIP
                file_list = zip_ref.namelist()
                
                # Look for supported data files
                data_files = [f for f in file_list if f.lower().endswith(('.csv', '.xlsx', '.xls', '.json'))]
                
                if not data_files:
                    st.error("❌ No CSV, Excel, or JSON files found in ZIP archive.")
                    return None
                
                if len(data_files) > 1:
                    st.info(f"📦 ZIP contains {len(data_files)} data files. Loading first one: {data_files[0]}")
                
                # Load the first data file found
                with zip_ref.open(data_files[0]) as file:
                    if data_files[0].lower().endswith('.csv'):
                        return pd.read_csv(file)
                    elif data_files[0].lower().endswith(('.xlsx', '.xls')):
                        return pd.read_excel(file)
                    elif data_files[0].lower().endswith('.json'):
                        return pd.read_json(file)
        else:
            st.error("❌ Unsupported file format. Use CSV, Excel, JSON, or ZIP.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def show_dataset_stats(df):
    """Display dataset statistics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Rows", f"{len(df):,}")
    with col2:
        st.metric("📋 Columns", df.shape[1])
    with col3:
        st.metric("❌ Missing", df.isnull().sum().sum())
    with col4:
        st.metric("🔢 Numeric", len(df.select_dtypes(include=[np.number]).columns))
    with col5:
        st.metric("📝 Categorical", len(df.select_dtypes(include=['object']).columns))

def handle_missing_values(df, strategy):
    """Handle missing values with different strategies"""
    df = df.copy()
    
    if strategy == "Drop rows":
        return df.dropna()
    elif strategy == "Fill with Mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        return df
    elif strategy == "Fill with Median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        return df
    elif strategy == "Fill with Mode":
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True) if len(df[col].mode()) > 0 else None
        return df
    elif strategy == "Forward Fill":
        return df.fillna(method='ffill').fillna(method='bfill')
    elif strategy == "Backward Fill":
        return df.fillna(method='bfill').fillna(method='ffill')
    return df

def encode_categorical(df, columns, method):
    """Encode categorical variables"""
    df = df.copy()
    if method == "Label Encoding":
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            st.session_state.label_encoders[col] = le
    elif method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df

def scale_features(df, columns, method):
    """Scale numeric features"""
    df = df.copy()
    if method == "Standardization":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df[columns] = scaler.fit_transform(df[columns])
    st.session_state.scaler = scaler
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_bar_chart(df, x_col, y_col=None):
    """Create bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    if y_col:
        df.groupby(x_col)[y_col].sum().plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    else:
        df[x_col].value_counts().plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(f"Bar Chart", fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col if y_col else 'Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_line_chart(df, x_col, y_col):
    """Create line chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df.sort_values(x_col) if pd.api.types.is_numeric_dtype(df[x_col]) else df
    ax.plot(df_sorted[x_col], df_sorted[y_col], marker='o', linewidth=2, markersize=6, color='steelblue')
    ax.set_title(f"Line Chart", fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_scatter(df, x_col, y_col):
    """Create scatter plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50, color='steelblue', edgecolors='black')
    ax.set_title(f"Scatter Plot", fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_histogram(df, col, bins=30):
    """Create histogram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df[col].dropna(), bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f"Histogram: {col}", fontsize=14, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Matrix Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("✨ SAFElytics")
st.sidebar.markdown("**Zero Code. Full Power.**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate:",
    ["📊 Home", "📁 Data Upload", "🔍 Data Preview", "⚙️ Processing", 
     "📈 Visualization", "🧠 Model Training", "💾 Download"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **No-Code ML Platform**
    
    Build ML models without writing code!
    - Upload data in any format
    - Process & explore data visually
    - Train models with clicks
    - Download results
""")

if st.sidebar.button("🔄 Reset All"):
    st.session_state.df = None
    st.session_state.df_processed = None
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.label_encoders = {}
    st.rerun()

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "📊 Home":
    st.title("✨ SAFElytics - Zero Code. Full Power.")
    st.markdown("*Developed by SAFE AI-AFRICA*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Data Status", "Ready" if st.session_state.df is not None else "Empty")
    with col2:
        st.metric("🔧 Processing", "Done" if st.session_state.df_processed is not None else "Pending")
    with col3:
        st.metric("🎯 Model", "Trained" if st.session_state.model is not None else "Empty")
    
    st.markdown("---")
    st.subheader("Quick Start Guide")
    st.markdown("""
    1. **📁 Data Upload**: Load your CSV, Excel, or JSON file
    2. **🔍 Data Preview**: Explore your data with sorting & filtering
    3. **⚙️ Processing**: Handle missing values, encode categories, scale features
    4. **📈 Visualization**: Create interactive charts
    5. **🧠 Model Training**: Build classification, regression, or clustering models
    6. **💾 Download**: Export processed data & trained models
    
    **Supported Models:**
    - Classification: Logistic Regression, Random Forest, SVM, KNN
    - Regression: Linear Regression, Random Forest Regressor
    - Clustering: KMeans
    """)

# ============================================================================
# PAGE: DATA UPLOAD
# ============================================================================
elif page == "📁 Data Upload":
    st.title("📁 Data Upload")
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json', 'zip'])
    
    if uploaded_file is not None:
        df = load_dataset(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.df_processed = df.copy()
            st.success(f"✅ File uploaded! Shape: {df.shape}")
            
            show_dataset_stats(df)
            
            st.subheader("Preview")
            rows = st.slider("Rows:", 1, len(df), 5)
            st.dataframe(df.head(rows), use_container_width=True)

# ============================================================================
# PAGE: DATA PREVIEW
# ============================================================================
elif page == "🔍 Data Preview":
    st.title("🔍 Data Preview")
    
    if st.session_state.df is None:
        st.warning("⚠️ Upload data first!")
    else:
        show_dataset_stats(st.session_state.df)
        
        st.subheader("Dataset")
        col1, col2 = st.columns(2)
        with col1:
            sort_col = st.selectbox("Sort by:", st.session_state.df.columns)
        with col2:
            sort_order = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)
        
        df_sorted = st.session_state.df.sort_values(
            sort_col, 
            ascending=(sort_order == "Ascending")
        )
        st.dataframe(df_sorted, use_container_width=True)
        
        st.subheader("Statistics")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        st.subheader("Data Info")
        info_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Type': st.session_state.df.dtypes,
            'Non-Null': st.session_state.df.count(),
            'Null': st.session_state.df.isnull().sum()
        })
        st.dataframe(info_df, use_container_width=True)

# ============================================================================
# PAGE: DATA PROCESSING
# ============================================================================
elif page == "⚙️ Processing":
    st.title("⚙️ Data Processing")
    
    if st.session_state.df is None:
        st.warning("⚠️ Upload data first!")
    else:
        st.session_state.df_processed = st.session_state.df.copy()
        
        # Missing Values
        st.subheader("1️⃣ Handle Missing Values")
        missing_cols = st.session_state.df_processed.columns[
            st.session_state.df_processed.isnull().any()
        ].tolist()
        
        if missing_cols:
            strategy = st.selectbox(
                "Strategy:",
                ["Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill", "Backward Fill"]
            )
            if st.button("Apply"):
                st.session_state.df_processed = handle_missing_values(st.session_state.df_processed, strategy)
                st.success(f"✅ Applied {strategy}")
        else:
            st.success("✅ No missing values")
        
        st.markdown("---")
        
        # Categorical Encoding
        st.subheader("2️⃣ Encode Categorical Variables")
        cat_cols = st.session_state.df_processed.select_dtypes(include=['object']).columns.tolist()
        
        if cat_cols:
            cols_to_encode = st.multiselect("Select columns:", cat_cols)
            method = st.radio("Method:", ["Label Encoding", "One-Hot Encoding"], horizontal=True)
            
            if st.button("Apply Encoding"):
                st.session_state.df_processed = encode_categorical(
                    st.session_state.df_processed, cols_to_encode, method
                )
                st.success(f"✅ Applied {method}")
        else:
            st.info("No categorical columns")
        
        st.markdown("---")
        
        # Feature Scaling
        st.subheader("3️⃣ Scale Numeric Features")
        num_cols = st.session_state.df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        if num_cols:
            cols_to_scale = st.multiselect("Select columns:", num_cols, key="scale_key")
            method = st.radio("Method:", ["Standardization", "Normalization"], horizontal=True)
            
            if st.button("Apply Scaling"):
                st.session_state.df_processed = scale_features(
                    st.session_state.df_processed, cols_to_scale, method
                )
                st.success(f"✅ Applied {method}")
        else:
            st.info("No numeric columns")
        
        st.markdown("---")
        
        # Feature Selection
        st.subheader("4️⃣ Feature Selection")
        action = st.radio("Action:", ["Keep All", "Remove Columns", "Remove Duplicates"], horizontal=True)
        
        if action == "Remove Columns":
            cols = st.multiselect("Select columns to remove:", st.session_state.df_processed.columns)
            if st.button("Remove"):
                st.session_state.df_processed = st.session_state.df_processed.drop(columns=cols)
                st.success("✅ Columns removed")
        
        elif action == "Remove Duplicates":
            if st.button("Remove Duplicates"):
                before = len(st.session_state.df_processed)
                st.session_state.df_processed = st.session_state.df_processed.drop_duplicates()
                st.success(f"✅ Removed {before - len(st.session_state.df_processed)} duplicates")
        
        st.markdown("---")
        st.subheader("Processed Data")
        st.dataframe(st.session_state.df_processed.head(), use_container_width=True)
        st.info(f"Shape: {st.session_state.df_processed.shape}")

# ============================================================================
# PAGE: VISUALIZATION
# ============================================================================
elif page == "📈 Visualization":
    st.title("📈 Visualization")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Process data first!")
    else:
        df = st.session_state.df_processed
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        chart_type = st.selectbox(
            "Chart Type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Correlation Heatmap"]
        )
        
        try:
            if chart_type == "Bar Chart":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X:", cat_cols if cat_cols else num_cols)
                with col2:
                    y_col = st.selectbox("Y:", num_cols)
                if st.button("Generate"):
                    fig = plot_bar_chart(df, x_col, y_col)
                    st.pyplot(fig)
            
            elif chart_type == "Line Chart":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X:", num_cols)
                with col2:
                    y_col = st.selectbox("Y:", num_cols)
                if st.button("Generate"):
                    fig = plot_line_chart(df, x_col, y_col)
                    st.pyplot(fig)
            
            elif chart_type == "Scatter Plot":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X:", num_cols)
                with col2:
                    y_col = st.selectbox("Y:", num_cols)
                if st.button("Generate"):
                    fig = plot_scatter(df, x_col, y_col)
                    st.pyplot(fig)
            
            elif chart_type == "Histogram":
                col = st.selectbox("Column:", num_cols)
                bins = st.slider("Bins:", 5, 100, 30)
                if st.button("Generate"):
                    fig = plot_histogram(df, col, bins)
                    st.pyplot(fig)
            
            elif chart_type == "Correlation Heatmap":
                if st.button("Generate"):
                    fig = plot_correlation_heatmap(df)
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE: MODEL TRAINING
# ============================================================================
elif page == "🧠 Model Training":
    st.title("🧠 Model Training")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Process data first!")
    else:
        df = st.session_state.df_processed
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select target
        st.subheader("Step 1: Target Variable")
        target = st.selectbox("Target Column:", df.columns)
        
        # Select problem type
        st.subheader("Step 2: Problem Type")
        problem = st.radio("Type:", ["Classification", "Regression", "Clustering"], horizontal=True)
        
        # Train-test split
        st.subheader("Step 3: Configuration")
        test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
        
        if st.button("🚀 Train Model"):
            try:
                X = df.drop(columns=[target]).select_dtypes(include=[np.number])
                y = df[target]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                if problem == "Classification":
                    model_name = st.selectbox("Model:", ["Logistic Regression", "Random Forest", "SVM", "KNN"])
                    
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100)
                    elif model_name == "SVM":
                        model = SVC()
                    else:
                        model = KNeighborsClassifier()
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    st.session_state.model = model
                    st.session_state.model_type = "Classification"
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("Precision", f"{precision:.4f}")
                    with col3:
                        st.metric("Recall", f"{recall:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{f1:.4f}")
                    
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    st.pyplot(fig)
                    
                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, y_pred))
                
                elif problem == "Regression":
                    model_name = st.selectbox("Model:", ["Linear Regression", "Random Forest Regressor"])
                    
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                    else:
                        model = RandomForestRegressor(n_estimators=100)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    st.session_state.model = model
                    st.session_state.model_type = "Regression"
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R² Score", f"{r2:.4f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with col3:
                        st.metric("MAE", f"{mae:.4f}")
                    with col4:
                        st.metric("MSE", f"{mse:.4f}")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, y_pred, alpha=0.6)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('Predicted vs Actual')
                    st.pyplot(fig)
                
                else:  # Clustering
                    n_clusters = st.slider("Clusters:", 2, 10, 3)
                    
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = model.fit_predict(X)
                    
                    st.session_state.model = model
                    st.session_state.model_type = "Clustering"
                    
                    st.metric("Inertia", f"{model.inertia_:.4f}")
                    
                    if X.shape[1] >= 2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)
                        ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], 
                                  c='red', s=200, marker='X', edgecolors='black')
                        plt.colorbar(scatter, ax=ax)
                        st.pyplot(fig)
                
                st.success("✅ Model trained!")
                st.session_state.model_metrics = {'trained': True}
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE: DOWNLOAD
# ============================================================================
elif page == "💾 Download":
    st.title("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Processed Data")
        if st.session_state.df_processed is not None:
            # CSV
            csv = st.session_state.df_processed.to_csv(index=False)
            st.download_button(
                "📄 CSV",
                csv,
                f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
            
            # Excel
            excel_buffer = io.BytesIO()
            st.session_state.df_processed.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            st.download_button(
                "📊 Excel",
                excel_buffer.getvalue(),
                f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("No data available")
    
    with col2:
        st.subheader("Trained Model")
        if st.session_state.model is not None:
            model_buffer = io.BytesIO()
            joblib.dump(st.session_state.model, model_buffer)
            model_buffer.seek(0)
            st.download_button(
                "🤖 Model",
                model_buffer.getvalue(),
                f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                "application/octet-stream"
            )
        else:
            st.info("No model available")
    
    st.markdown("---")
    st.subheader("Preprocessing Objects")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.scaler is not None:
            scaler_buffer = io.BytesIO()
            joblib.dump(st.session_state.scaler, scaler_buffer)
            scaler_buffer.seek(0)
            st.download_button(
                "🔧 Scaler",
                scaler_buffer.getvalue(),
                f"scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                "application/octet-stream"
            )
    
    with col2:
        if st.session_state.label_encoders:
            encoders_buffer = io.BytesIO()
            joblib.dump(st.session_state.label_encoders, encoders_buffer)
            encoders_buffer.seek(0)
            st.download_button(
                "🏷️ Encoders",
                encoders_buffer.getvalue(),
                f"encoders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                "application/octet-stream"
            )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p><strong>✨ SAFElytics</strong> | Zero Code. Full Power.</p>
    <p>Developed by <strong>SAFE AI-AFRICA</strong> | <a href='https://safeai-africa.com' target='_blank'>safeai-africa.com</a></p>
    <p>Powered by Streamlit, Scikit-learn, Pandas | Version 1.0</p>
    </div>
""", unsafe_allow_html=True)
