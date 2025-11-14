# 📊 Sample Datasets for Testing

You can create these sample CSV files to test the application.

## 1. Iris Dataset (Classification)
Create a file named `iris.csv`:

```
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
4.7,3.2,1.3,0.2,setosa
6.4,3.2,4.5,1.5,versicolor
7.1,3.0,5.9,2.1,virginica
4.6,3.1,1.5,0.2,setosa
5.9,3.0,4.2,1.5,versicolor
6.7,3.0,5.0,1.7,versicolor
```

## 2. House Prices Dataset (Regression)
Create a file named `house_prices.csv`:

```
square_feet,bedrooms,bathrooms,age,price
1500,3,2,10,300000
2000,4,2.5,5,400000
1200,2,1,20,250000
2500,4,3,2,500000
1800,3,2,8,350000
2200,4,3,1,450000
1000,2,1,30,200000
2100,4,2.5,3,420000
1600,3,2,12,310000
2300,4,3,0,480000
```

## 3. Customer Segmentation Dataset (Clustering)
Create a file named `customers.csv`:

```
age,income,spending
25,30000,5000
35,50000,15000
45,60000,18000
55,70000,20000
28,35000,7000
38,55000,16000
48,65000,19000
58,75000,22000
26,32000,6000
36,52000,14000
```

## 4. Employee Dataset (Mixed)
Create a file named `employees.csv`:

```
age,salary,years_experience,department,performance_rating
25,40000,1,Sales,3.5
35,55000,10,IT,4.2
45,65000,20,Finance,4.0
28,45000,3,Sales,3.8
38,60000,12,IT,4.5
48,70000,22,Finance,4.3
26,42000,2,Sales,3.6
36,58000,11,IT,4.1
46,68000,21,Finance,4.2
```

## 5. Medical Dataset (Classification with Missing Values)
Create a file named `medical.csv`:

```
age,blood_pressure,cholesterol,glucose,diabetes
45,120,200,100,0
55,140,220,120,1
35,110,180,90,0
65,150,240,140,1
50,130,210,110,1
40,115,190,95,0
60,145,230,130,1
48,125,205,105,0
58,135,225,125,1
52,128,208,112,1
```

## How to Use These Files

1. **Copy any CSV content** from above
2. **Paste into a text editor** (Notepad, VS Code, etc.)
3. **Save as `.csv`** (e.g., `iris.csv`)
4. **Place in `c:\Users\USER\Desktop\streamlit\`** folder
5. **Open the app** and upload the file

## Testing Recommendations

### 1. Iris Dataset (Best for starting)
- **Use case**: Classification
- **Features**: 4 numeric features (sepal/petal measurements)
- **Target**: Species (3 classes)
- **Steps**: 
  1. Upload → Preview
  2. No preprocessing needed (already clean)
  3. Visualize with scatter plots
  4. Train classification model
  5. Compare different models

### 2. House Prices (Regression testing)
- **Use case**: Price prediction
- **Features**: 4 numeric features
- **Target**: Price (numeric)
- **Steps**:
  1. Upload → Preview
  2. Scale features (optional)
  3. Create scatter plots for insights
  4. Train regression model
  5. Compare Linear vs Random Forest

### 3. Customer Segmentation (Clustering testing)
- **Use case**: Customer groups
- **Features**: 3 numeric features
- **No target** column needed
- **Steps**:
  1. Upload → Preview
  2. Scale features
  3. Use clustering model
  4. Try different cluster counts (2-5)

### 4. Employee Dataset (Mixed workflow)
- **Use case**: Full workflow demo
- **Features**: Mix of numeric and categorical
- **Steps**:
  1. Upload → Preview
  2. Handle any missing values
  3. Encode department if needed
  4. Scale numeric features
  5. Create visualizations
  6. Build predictive model

### 5. Medical Dataset (Error handling practice)
- **Use case**: Test error handling
- **Features**: Numeric with some missing values
- **Steps**:
  1. Upload
  2. Practice missing value strategies
  3. Train model with cleaned data

## 📊 Creating Your Own Dataset

**CSV Format Example:**
```
feature1,feature2,target
value1,value2,value3
value4,value5,value6
```

**Requirements:**
- First row = column names
- Comma-separated values
- No special characters in headers
- Consistent data types per column

## 🧪 Testing Checklist

Using any sample dataset:

- [ ] Upload the CSV file successfully
- [ ] View dataset in preview
- [ ] Check statistics display correctly
- [ ] Sort by different columns
- [ ] Create visualizations (all types)
- [ ] Handle missing values (if needed)
- [ ] Scale features
- [ ] Train classification model
- [ ] View metrics and confusion matrix
- [ ] Download processed data
- [ ] Download trained model

## 💡 Tips for Testing

1. **Start small**: Use Iris (150 samples)
2. **Test incrementally**: One feature at a time
3. **Try different models**: Compare results
4. **Use visualizations**: Understand your data
5. **Download often**: Save your work

## 🔗 Finding More Datasets

Popular sources for ML datasets:
- **Kaggle**: kaggle.com (thousands of datasets)
- **UCI ML Repository**: archive.ics.uci.edu
- **Google Dataset Search**: datasetsearch.research.google.com
- **GitHub**: Search "csv dataset"

## ✅ Validation

Each dataset should work without errors in the app:
1. Upload successfully ✅
2. Preview displays ✅
3. Statistics calculate ✅
4. Visualizations generate ✅
5. Models train ✅
6. Results download ✅

---

**Start testing with these samples and explore the app's capabilities!**
