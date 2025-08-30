# 🌿 Crop Leaf Disease Classification with CNN and Cross-Validation

This repository contains experiments on **Convolutional Neural Networks (CNNs)** for crop leaf disease classification. 
Explarong more on my CNN model , i have used the K-Cross validation methods , i.e the 5-fold and 10-fold. Apart from my previous research in which in bahve used the usually train,test and spilt method , but running on the K-Fold Cross Validation method had provided a good study of the dataset preprocessing and also for the CNN model.

The focus is on evaluating the effect of **cross-validation strategies (5-fold, 10-fold)** and analyzing model performance.  

I have also apply **systematic preprocessing** and build a **custom combined dataset** from multiple public sources (Kaggle/PlantVillage).

## 🗂️ Dataset Construction

The dataset was **not directly downloaded as a single set**.  
Instead, it was **constructed by merging multiple plant disease datasets** from Kaggle and PlantVillage.

Steps to make the dataset:

1. **Collect datasets**  
   - Tomato leaf diseases (e.g., Early Blight, Leaf Mold, Target Spot, etc.)  
   - Rice leaf diseases  
   - Cotton leaf diseases  
   - Healthy leaves for each crop  

2. **Unify the structure**  
   - All images were placed in a **common folder hierarchy**.  
   - Each image was renamed or stored under a directory representing its **class label** (e.g., `Tomato_Early_blight/001.jpg`).  

3. **Generate CSV file**  
   - A master CSV file `combined_crop_disease_labels.csv` was created with two columns:  

     ```csv
     image,label
     Tomato_Early_blight/001.jpg,Tomato_Early_blight
     Tomato_Leaf_Mold/002.jpg,Tomato_Leaf_Mold
     Cotton_Leaf_Spot/003.jpg,Cotton_Leaf_Spot
     ...
     ```

   - `image` → relative path to the image  
   - `label` → class name (`Crop_Disease`)  

4. **Label Encoding**  
   - A new column `label_encoded` was added using `pandas.factorize()`.  
   - A `label_str` column was also created for use with `ImageDataGenerator`.  

✅ This unified dataset allows us to apply **Stratified Cross-Validation** across crops & diseases.

---

## 🔄 Preprocessing Pipeline

Before training CNN models, the following preprocessing was applied:

1. **Resize**: All images resized to `256 × 256`.  
2. **Normalize**: Pixel values scaled to `[0, 1]`.  
3. **Augmentation**:  
   - Random rotations (±20°)  
   - Width & height shifts  
   - Shear, zoom  
   - Horizontal flips  

   Implemented via `ImageDataGenerator`.  
4. **Cross-validation splits**:  
   - 5-fold (`5-fold.ipynb`, `5-fold-validation.ipynb`)  
   - 10-fold (`10-fold-validation.ipynb`)  
   - StratifiedKFold ensures balanced class distribution.  

---

## 🧪 Notebooks Overview

- **5-fold.ipynb**  
  Full 5-fold cross-validation with:
  - ROC curve calculation for each fold & class  
  - Model saving (`.keras` per fold)  

- **5-fold-validation.ipynb**  
  Simplified 5-fold CV:
  - Faster to run  
  - No ROC curves, no model saving  

- **10-fold-validation.ipynb**  
  Same as above but with **10 folds** for more robust evaluation.  

- **analysis.ipynb**  
  Plots training curves, confusion matrices, and final evaluation metrics.  

