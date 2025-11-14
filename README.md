# 🤖 SAFElytics - No-Code Machine Learning Tool

A comprehensive, user-friendly **Streamlit web application** for machine learning without writing code. Perfect for data analysts, business users, and ML enthusiasts who want to build, train, and evaluate machine learning models with just a few clicks.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 🌟 Features

### 📊 Data Import & Exploration
- ✅ Upload datasets in **multiple formats**: CSV, Excel (.xlsx), JSON
- ✅ Interactive dataset preview with **sorting and filtering**
- ✅ Comprehensive statistics and data type analysis
- ✅ Missing value detection and visualization
- ✅ Column-level statistics (unique values, null counts, memory usage)

### 🔧 Data Processing
- ✅ **Missing Value Handling** (7+ strategies):
  - Drop rows (any/all missing values)
  - Fill with Mean/Median/Mode
  - Forward/Backward fill
  
- ✅ **Categorical Encoding**:
  - Label Encoding
  - One-Hot Encoding
  
- ✅ **Feature Scaling**:
  - Standardization (Z-score normalization)
  - Min-Max Normalization
  
- ✅ **Feature Selection**:
  - Remove specific columns
  - Remove duplicate rows
  - Keep/manage feature set

### 📈 Visualization & EDA
Create publication-quality visualizations with one click:
- 📊 **Bar Charts** - Categorical distributions
- 📈 **Line Charts** - Trends over time
- 🔵 **Scatter Plots** - Relationships between variables
- 📊 **Histograms** - Numeric distributions
- 🔥 **Correlation Heatmaps** - Feature correlations

### 🧠 Machine Learning Models
Build ML models without writing a single line of code!

#### Classification
- 🔹 Logistic Regression
- 🌲 Random Forest Classifier
- 🎯 Support Vector Machine (SVM)
- 👥 K-Nearest Neighbors (KNN)

#### Regression
- 📊 Linear Regression
- 🌲 Random Forest Regressor

#### Clustering
- 🎯 K-Means Clustering

### 📊 Model Evaluation & Metrics
- ✅ **Classification**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report
- ✅ **Regression**: R² Score, RMSE, MAE, MSE
- ✅ **Clustering**: Inertia, Silhouette Score
- ✅ Visual plots for model performance

### 💾 Export & Download
- 📥 Download processed datasets (CSV, Excel, JSON)
- 📥 Download trained models (.pkl files)
- 📥 Download scalers and label encoders
- 📥 Model performance metrics summaries

### 🎨 User Interface
- ✅ **Clean, intuitive design** with sidebar navigation
- ✅ **Responsive layout** that works on all screen sizes
- ✅ **Dark-friendly styling** for better readability
- ✅ **Comprehensive error handling** with helpful messages
- ✅ **Session state management** for seamless user experience

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/SAFElytics.git
cd SAFElytics
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📚 Usage Guide

### Step 1: Upload Your Data
1. Go to **📊 Data Upload** tab
2. Click "Browse files" and select your CSV, Excel, or JSON file
3. Your data will be automatically loaded and previewed

### Step 2: Explore Your Data
1. Go to **🔍 Data Preview** tab
2. View dataset statistics and information
3. Sort and filter columns as needed
4. Identify missing values and data types

### Step 3: Process Your Data
1. Go to **⚙️ Data Processing** tab
2. Handle missing values using your preferred strategy
3. Encode categorical variables
4. Scale numeric features
5. Select relevant features

### Step 4: Visualize Your Data
1. Go to **📈 Visualization** tab
2. Choose visualization type (bar, line, scatter, histogram, or heatmap)
3. Select columns to visualize
4. Click "Generate [Chart Type]"

### Step 5: Build ML Models
1. Go to **🧠 Model Training** tab
2. Select target variable
3. Choose problem type (Classification/Regression/Clustering)
4. Set train/test split ratio
5. Choose and train your model
6. View comprehensive evaluation metrics

### Step 6: Download Results
1. Go to **💾 Download Results** tab
2. Download processed datasets in your preferred format
3. Download trained models for future use
4. Download scalers and encoders for preprocessing consistency

## 📋 Project Structure

```
SAFElytics/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
├── GITHUB_SETUP.md            # GitHub setup instructions
├── QUICK_START.md             # Quick start guide
├── PROJECT_SUMMARY.md         # Project overview
└── SAMPLE_DATASETS.md         # Sample datasets information
```

## 🔧 Technologies Used

### Core Framework
- **Streamlit** 1.28+ - Web app framework
- **Python** 3.8+ - Programming language

### Data Science Libraries
- **Pandas** 2.0+ - Data manipulation
- **NumPy** 1.24+ - Numerical computing
- **Scikit-learn** 1.3+ - Machine learning
- **Matplotlib** 3.7+ - Visualization
- **Seaborn** 0.12+ - Statistical visualization
- **Joblib** 1.3+ - Model serialization

### Additional Libraries
- **OpenPyXL** 3.1+ - Excel file support

## 📊 Example Workflow

```
1. Upload sales_data.csv
   ↓
2. Explore data (500 rows, 8 columns)
   ↓
3. Handle missing values in "Price" column (fill with median)
   ↓
4. Encode "Category" as one-hot
   ↓
5. Scale numeric features (standardization)
   ↓
6. Visualize Price vs Sales correlation
   ↓
7. Train Random Forest Regressor to predict Sales
   ↓
8. Achieve R² = 0.92 on test set
   ↓
9. Download model and processed data
```

## 🎯 Use Cases

- **Sales Forecasting** - Predict future sales using historical data
- **Customer Segmentation** - Cluster customers for targeted marketing
- **Churn Prediction** - Identify at-risk customers
- **Price Optimization** - Build pricing models
- **Risk Assessment** - Classify high/low-risk scenarios
- **Student Performance** - Predict academic outcomes
- **Medical Diagnosis** - Support healthcare decisions
- **Market Analysis** - Identify trends and patterns

## ⚙️ Configuration

### Customization
The app can be customized by modifying `app.py`:

- **Add more ML models** - Extend the model selection
- **Custom visualizations** - Add new chart types
- **Additional preprocessing** - Implement more data transformations
- **Theme customization** - Modify CSS styling

## 🐛 Troubleshooting

### "streamlit: command not found"
```bash
pip install streamlit
```

### "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install -r requirements.txt
```

### Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

### Memory issues with large files
- Use smaller datasets for testing
- Consider data sampling for initial exploration

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit (`git commit -m 'Add AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- ML algorithms from [Scikit-learn](https://scikit-learn.org/)
- Visualizations with [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)

## 📞 Support & Contact

- 📧 Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/SAFElytics/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/YOUR_USERNAME/SAFElytics/discussions)

## 🗺️ Roadmap

- [ ] Add time series forecasting models (ARIMA, Prophet)
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add model comparison and AutoML features
- [ ] Implement cross-validation
- [ ] Add feature importance visualization
- [ ] Support for larger datasets with streaming
- [ ] Docker containerization
- [ ] Cloud deployment templates (AWS, GCP, Azure)

---

**Made with ❤️ for the ML Community**

**Last Updated:** November 14, 2025
Convert categorical variables to numeric:
- **Label Encoding**: Maps categories to integers (0, 1, 2, ...)
- **One-Hot Encoding**: Creates binary columns for each category

#### C. Feature Scaling
Normalize numeric features:
- **Standardization**: Z-score normalization (mean=0, std=1)
- **Normalization**: Min-Max scaling (0 to 1 range)

#### D. Feature Selection
- Remove specific columns
- Remove duplicate rows
- Keep all features

### 4. **Visualization** 📈

Create professional charts with a single click:

- **Bar Chart**: Compare categories and their values
- **Line Chart**: Visualize trends over ordered data
- **Scatter Plot**: Explore relationships between two variables
- **Histogram**: Analyze distributions of numeric variables
- **Correlation Heatmap**: Identify relationships between all numeric features

All charts are interactive and fully customizable.

### 5. **Model Training** 🧠

#### Classification Models
- **Logistic Regression**: Fast linear classifier
- **Random Forest**: Ensemble method with multiple decision trees
- **Support Vector Machine (SVM)**: Powerful kernel-based classifier
- **K-Nearest Neighbors (KNN)**: Instance-based learning

**Evaluation Metrics:**
- Accuracy: Overall correctness of predictions
- Precision: Correct positive predictions
- Recall: Proportion of actual positives found
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed classification results
- Classification Report: Comprehensive metrics per class

#### Regression Models
- **Linear Regression**: Simple linear relationship modeling
- **Random Forest Regressor**: Ensemble method for regression

**Evaluation Metrics:**
- R² Score: Proportion of variance explained
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- Actual vs Predicted scatter plot

#### Clustering
- **KMeans**: Partition data into K clusters

**Configuration:**
- Specify number of clusters (2-10)
- View inertia metric
- Visualize clusters in 2D space

### 6. **Download Results** 💾

Export your work in multiple formats:

**Processed Data:**
- CSV format
- Excel format (.xlsx)
- JSON format

**Trained Models:**
- Serialized model file (.pkl)
- Can be loaded for future predictions

**Preprocessing Objects:**
- Scaler objects (StandardScaler, MinMaxScaler)
- Label encoders for later encoding

## 📊 User Interface

### Navigation Structure
```
Sidebar Menu
├── 📊 Home (Dashboard overview)
├── 📁 Data Upload
├── 🔍 Data Preview
├── ⚙️ Processing
├── 📈 Visualization
├── 🧠 Model Training
└── 💾 Download
```

### Page Layout
- **Wide layout** for better data visualization
- **Responsive design** that adapts to different screen sizes
- **Interactive elements** with real-time updates
- **Color-coded messages** for user feedback

## 🛠️ Technical Details

### Technologies Used
- **Streamlit**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **Joblib**: Model serialization

### Session State Management
The app maintains state across reruns using Streamlit's session_state, allowing you to:
- Upload data once and use it throughout the session
- Build on previous preprocessing steps
- Compare different models
- Download results anytime

### Error Handling
- Comprehensive error messages
- Input validation
- Graceful failure handling
- User-friendly error notifications

## 📈 Workflow Example

1. **Upload Data**: Click "Data Upload" → Select your CSV/Excel file
2. **Explore**: View in "Data Preview" → Sort and filter columns
3. **Clean**: Go to "Processing" → Handle missing values
4. **Transform**: Still in "Processing" → Encode categories, scale features
5. **Analyze**: Click "Visualization" → Create charts
6. **Model**: Go to "Model Training" → Select problem type and model
7. **Evaluate**: View metrics and visualizations
8. **Export**: Use "Download" → Get your processed data and trained model

## 💡 Tips & Tricks

1. **Always preview first**: Use Data Preview before processing
2. **Keep copies**: Download your processed data for backup
3. **Test with small subsets**: Use fewer rows first to validate workflow
4. **Check correlations**: Use heatmap to identify important features
5. **Start simple**: Try Logistic Regression before complex models
6. **Use appropriate features**: Select numeric features for most models

## 🐛 Troubleshooting

**Issue**: "No numeric columns found"
- **Solution**: Encode categorical columns first in Processing tab

**Issue**: "ValueError: y only has 1 class"
- **Solution**: Your target variable needs at least 2 different values for classification

**Issue**: Charts not generating
- **Solution**: Ensure you have numeric columns selected for chart axes

**Issue**: Model training fails
- **Solution**: Check that your target column isn't dropped from features (X)

## 📝 File Formats

### CSV
```
feature1,feature2,target
1.5,2.3,0
2.1,3.4,1
```

### Excel (.xlsx)
- First row contains headers
- Each column is a feature
- No special formatting needed

### JSON
```json
[
  {"feature1": 1.5, "feature2": 2.3, "target": 0},
  {"feature1": 2.1, "feature2": 3.4, "target": 1}
]
```

## 🎯 Best Practices

1. **Data Quality**: Ensure your data is clean before uploading
2. **Feature Selection**: Remove irrelevant columns
3. **Train-Test Split**: Default 20% test size is reasonable
4. **Model Selection**: Choose based on your problem type
5. **Evaluation**: Always check multiple metrics
6. **Reproducibility**: Save your models for later use

## 📞 Support

For issues or feature requests, check that all dependencies are installed:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn openpyxl joblib
```

## 📜 License

This application is free to use and modify for personal and commercial projects.

---

**Created with ❤️ for machine learning enthusiasts**
**Version 1.0 | 2025**
