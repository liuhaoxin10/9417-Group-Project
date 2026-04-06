## 2. Data Sources

- Wine: Built-in dataset from sklearn  
- Divorce: UCI Machine Learning Repository  
- German Credit: UCI Machine Learning Repository  
- Bike Sharing: UCI Bike Sharing Dataset  
- Appliances Energy: UCI Appliances Energy Prediction Dataset  

数据来源：

- Wine：sklearn 内置数据集  
- Divorce：UCI 机器学习库  
- German Credit：UCI 机器学习库  
- Bike Sharing：UCI Bike Sharing 数据集  
- Appliances Energy：UCI Appliances Energy 数据集  

---

## 3. Raw Data Files

All raw datasets are stored in:


data/raw/


主要文件包括：

- divorce.csv  
- german.data  
- day.csv  
- energydata_complete.csv  

---

## 4. Processed Data Files
我放了两个版本的预处理代码：一个是会保存处理后数据的版本和一个不保存数据直接输出的版本
处理后的数据在data/processed。

---

## 5. Data Storage Locations

- Raw data:  

data/raw/


- Processed data:  
in data/processed
- Processed data:

data/processed/

Processed datasets are stored as separate files, including:

- day_processed.csv  
- divorce_processed.csv  
- energydata_complete_processed.csv  
- german_processed.data  
- wine_processed.data  

The processed data includes cleaned and feature-engineered versions of the raw datasets.
- Outputs (results):  

outputs/tables/


---

## 6. Processing Script

The main script used for data preprocessing and modeling is:


experiments/dataset1/xrfm_pipeline.py


---

## 7. Preprocessing Steps

The following preprocessing steps are applied:

- Missing value handling (median / most frequent imputation)  
- One-hot encoding for categorical variables  
- Standardization of numerical features  
- Date feature extraction (hour, day, month, weekday)  
- Removal of redundant features (e.g., casual, registered in bike dataset)  
- Train / validation / test split  

主要预处理步骤包括：

- 缺失值填充（中位数 / 众数）  
- 类别特征 One-hot 编码  
- 数值特征标准化  
- 时间特征提取（小时、日期、月份、星期）  
- 删除冗余特征（如 bike 数据中的 casual、registered）  
- 训练集 / 验证集 / 测试集划分  

---

## 8. Reproducibility Instructions

To reproduce the results:

1. Install dependencies:

pip install -r requirements.txt


2. Ensure all raw data files are placed under:

data/raw/


3. Run the main script:

python experiments/dataset1/xrfm_pipeline.py


4. The results will be generated in:

outputs/tables/


---

## 9. Notes

- The Appliances Energy dataset is sampled to n = 10000 to meet assignment requirements  
- xRFM may fail on large datasets due to memory limitations  
- A fallback model (HistGradientBoostingRegressor) is used to ensure successful execution  

说明：

- Appliances Energy 数据集采样为 n = 10000 以满足作业要求  
- xRFM 在大规模数据上可能因内存限制失败  
- 使用备用模型保证实验可以完成  