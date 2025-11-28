# Resource-Efficient Knowledge Distillation for Multi-Domain Time Series Forecasting

This repository contains the implementation of our project **“Resource Efficient Knowledge Distillation from Multiple Domains in Time Series Forecasting”**, developed as part of IE643: Deep Learning (IIT Bombay). The goal of this project is to compress a large, high-performance **Autoformer** teacher model into a lightweight **TSMixer** student while maintaining strong forecasting accuracy across multiple heterogeneous electricity datasets.

We explore several novel knowledge distillation (KD) strategies—including **adaptive alpha**, **alpha smoothing**, **learnable alpha**, **projected multi-dataset distillation**, and **intermediate feature distillation**—to improve cross-domain transfer, training stability, and resource efficiency. The repository includes complete training pipelines for teacher models, multi-dataset KD, feature projection, and student evaluation.

---

## 📁 Datasets Used

All four electricity consumption datasets used in this project are available in a consolidated Google Drive folder:

👉 **Download here:**  
https://drive.google.com/drive/folders/1TH3lVjDuTfw9rzhwe-vy8TWwkh657vI-?usp=sharing

Below is a brief overview of each dataset:

---

### **1. Electricity Countrywise**
**Source:** https://data.open-power-system-data.org/time_series/  
**Unit:** Megawatt (MW)  
**Frequency:** Hourly  

This dataset contains country-level electricity consumption aggregated by TSOs and power exchanges via ENTSO-E Transparency, covering EU and neighboring regions (2015–2020).

**Data Cleaning:**  
- Reduced irrelevant columns (300 → 57)  
- Standardized timestamps by keeping only the UTC column  
- Handled missing values using interpolation + forward/backward fill  

---

### **2. Electricity Household**
**Source:** https://github.com/thuml/Autoformer?tab=readme-ov-file  
**Unit:** Kilowatt (kW)  
**Frequency:** Hourly (96 measurements/day due to 15-min intervals)  

Each column represents an individual client’s electricity usage, with DST adjustments based on Portuguese time.

**Data Cleaning:**  
- Already cleaned in the official Autoformer repository; no additional preprocessing required  

---

### **3. Electricity Areawise**
**Source:**  
https://www.aeso.ca/market/market-and-system-reporting/data-requests/hourly-load-by-area-and-region  
**Unit:** Megawatt (MW)  
**Frequency:** Hourly  

Contains hourly load data by area and region from January 2011 to December 2024.

**Data Cleaning:**  
- Removed non-load metadata columns (e.g., North, East)  
- Dropped the first informational row  

---

### **4. Electricity Jerico**
**Source:**  
https://springernature.figshare.com/articles/dataset/JERICHO-E-usage_dataset/13456355  
**Unit:** Kilowatt (kW)  
**Frequency:** Hourly  

Represents electricity consumption for 38 NUTS2 regions in Germany, aggregated across residential, industrial, commercial, and mobility sectors.

**Data Cleaning:**  
- Combined multiple sector-wise files by summing corresponding consumption columns  

---

