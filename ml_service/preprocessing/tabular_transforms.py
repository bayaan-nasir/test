"""
preprocessing/tabular_transforms.py
Feature engineering and input validation for tabular (structured) data models.
Used by Diabetes, Cardiovascular, and other tabular-input disease models.

Key responsibilities:
  1. Validate that all required features are present in the input (with fallbacks)
  2. Replace biologically impossible zero values with NaN (for imputation)
  3. Return a clean numpy array in the correct feature order

Why zeros are treated as missing in PIMA:
  Several PIMA features (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
  cannot be zero in a living patient. These zeros are data entry artefacts
  representing missing values. The pipeline's SimpleImputer(median) handles them.
"""
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from core.logger import logger


#  DIABETES 

class DiabetesInput(BaseModel):
    """
    Patient data for Type 2 Diabetes prediction.
    All fields are validated for plausible clinical ranges.
    Django sends this JSON body to POST /api/ml/predict/diabetes
    """
    pregnancies:               float = Field(default=0.0, ge=0, le=20,  description="Number of pregnancies (0 for male patients)")
    glucose:                   float = Field(default=0.0, ge=0, le=300, description="Plasma glucose concentration (mg/dL)")
    blood_pressure:            float = Field(default=0.0, ge=0, le=200, description="Diastolic blood pressure (mmHg)")
    skin_thickness:            float = Field(default=0.0, ge=0, le=100, description="Triceps skinfold thickness (mm)")
    insulin:                   float = Field(default=0.0, ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    bmi:                       float = Field(default=0.0, ge=0, le=80,  description="Body mass index (kg/m²)")
    diabetes_pedigree_function: float = Field(default=0.0, ge=0, le=3,   description="Diabetes pedigree function (family history score)")
    age:                       float = Field(default=0.0, ge=0, le=120, description="Age in years")

    @field_validator("glucose", "blood_pressure", "bmi")
    @classmethod
    def critical_fields_not_zero(cls, v, info):
        """Warn if critical clinical values are zero — likely missing data."""
        if v == 0:
            logger.warning(
                f"Field '{info.field_name}' received value 0 — "
                "will be treated as missing and imputed."
            )
        return v

ZERO_AS_NAN_FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

DIABETES_FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

def preprocess_diabetes_input(data: DiabetesInput) -> np.ndarray:
    row = {
        "Pregnancies":              data.pregnancies,
        "Glucose":                  data.glucose,
        "BloodPressure":            data.blood_pressure,
        "SkinThickness":            data.skin_thickness,
        "Insulin":                  data.insulin,
        "BMI":                      data.bmi,
        "DiabetesPedigreeFunction": data.diabetes_pedigree_function,
        "Age":                      data.age,
    }
    for feature in ZERO_AS_NAN_FEATURES:
        if row[feature] == 0.0:
            row[feature] = np.nan
    values = [row[feat] for feat in DIABETES_FEATURE_ORDER]
    return np.array(values, dtype=np.float32).reshape(1, -1)

def load_diabetes_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Outcome" in df.columns:
        df = df.rename(columns={"Outcome": "target"})
    for col in ZERO_AS_NAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    logger.info(f"Loaded diabetes dataset: {len(df)} rows | class balance: {df['target'].value_counts().to_dict()}")
    return df


#  HEART DISEASE 

class HeartDiseaseInput(BaseModel):
    """
    Patient data for Heart Disease prediction.
    Field names follow the UCI Cleveland dataset convention.
    """
    age:      float = Field(default=0.0, ge=0, le=120, description="Age in years")
    sex:      int   = Field(default=0,   ge=0, le=1,   description="1 = male, 0 = female")
    cp:       int   = Field(default=0,   ge=0, le=3,   description="Chest pain type (0-3)")
    trestbps: float = Field(default=0.0, ge=0, le=250, description="Resting blood pressure (mmHg)")
    chol:     float = Field(default=0.0, ge=0, le=600, description="Serum cholesterol (mg/dl)")
    fbs:      int   = Field(default=0,   ge=0, le=1,   description="Fasting blood sugar > 120 mg/dl (1=yes, 0=no)")
    restecg:  int   = Field(default=0,   ge=0, le=2,   description="Resting ECG result (0-2)")
    thalach:  float = Field(default=0.0, ge=0, le=250, description="Maximum heart rate achieved")
    exang:    int   = Field(default=0,   ge=0, le=1,   description="Exercise-induced angina (1=yes, 0=no)")
    oldpeak:  float = Field(default=0.0, ge=0, le=10,  description="ST depression induced by exercise")
    slope:    int   = Field(default=0,   ge=0, le=2,   description="Slope of peak exercise ST segment (0-2)")
    ca:       int   = Field(default=0,   ge=0, le=4,   description="Number of major vessels coloured by fluoroscopy (0-3)")
    thal:     int   = Field(default=0,   ge=0, le=3,   description="Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)")

HEART_DISEASE_FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

def preprocess_heart_disease_input(data: HeartDiseaseInput) -> np.ndarray:
    row = {
        "age": data.age, "sex": data.sex, "cp": data.cp,
        "trestbps": data.trestbps, "chol": data.chol, "fbs": data.fbs,
        "restecg": data.restecg, "thalach": data.thalach, "exang": data.exang,
        "oldpeak": data.oldpeak, "slope": data.slope, "ca": data.ca, "thal": data.thal,
    }
    values = [row[feat] for feat in HEART_DISEASE_FEATURE_ORDER]
    return np.array(values, dtype=np.float32).reshape(1, -1)

def load_heart_disease_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename_map = {
        "thalch": "thalach", "thalachh": "thalach", "trtbps": "trestbps",
        "exng": "exang", "slp": "slope", "caa": "ca", "thall": "thal",
        "output": "target", "num": "target"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "sex" in df.columns and df["sex"].dtype == object:
        df["sex"] = df["sex"].map({"Male": 1, "Female": 0, "male": 1, "female": 0})
    if "cp" in df.columns and df["cp"].dtype == object:
        df["cp"] = df["cp"].map({"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "non-anginal pain": 2, "asymptomatic": 3})
    if "fbs" in df.columns and df["fbs"].dtype == object:
        df["fbs"] = df["fbs"].map({"True": 1, "False": 0, "true": 1, "false": 0, True: 1, False: 0})
    if "restecg" in df.columns and df["restecg"].dtype == object:
        df["restecg"] = df["restecg"].map({"normal": 0, "stt abnormality": 1, "ST-T wave abnormality": 1, "lv hypertrophy": 2, "left ventricular hypertrophy": 2})
    if "exang" in df.columns and df["exang"].dtype == object:
        df["exang"] = df["exang"].map({"True": 1, "False": 0, "true": 1, "false": 0, True: 1, False: 0})
    if "slope" in df.columns and df["slope"].dtype == object:
        df["slope"] = df["slope"].map({"upsloping": 0, "flat": 1, "downsloping": 2})
    if "thal" in df.columns and df["thal"].dtype == object:
        df["thal"] = df["thal"].map({"normal": 1, "fixed defect": 2, "reversable defect": 3, "reversible defect": 3})
    if "target" in df.columns:
        df["target"] = (df["target"] > 0).astype(int)

    df = df.replace("?", np.nan)
    features_to_cast = HEART_DISEASE_FEATURE_ORDER
    for col in features_to_cast:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Loaded heart disease dataset: {len(df)} rows")
    return df


#  ANAEMIA 

class AnaemiaInput(BaseModel):
    gender:      int   = Field(default=0,   ge=0, le=1,   description="1 = male, 0 = female")
    hemoglobin:  float = Field(default=0.0, ge=0, le=22,  description="Haemoglobin level (g/dL)")
    mch:         float = Field(default=0.0, ge=0, le=50,  description="Mean Corpuscular Haemoglobin (pg)")
    mchc:        float = Field(default=0.0, ge=0, le=45,  description="Mean Corpuscular Haemoglobin Concentration (g/dL)")
    mcv:         float = Field(default=0.0, ge=0, le=130, description="Mean Corpuscular Volume (fL)")

ANAEMIA_FEATURE_ORDER = ["Gender", "MCH", "MCHC", "MCV"]

def preprocess_anaemia_input(data: AnaemiaInput) -> np.ndarray:
    row = {
        "Gender":     data.gender,
        "MCH":        data.mch,
        "MCHC":       data.mchc,
        "MCV":        data.mcv,
    }
    values = [row[feat] for feat in ANAEMIA_FEATURE_ORDER]
    return np.array(values, dtype=np.float32).reshape(1, -1)

def load_anaemia_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Result" in df.columns:
        df = df.rename(columns={"Result": "target"})
    logger.info(f"Loaded anaemia dataset: {len(df)} rows")
    return df


#  HYPERTENSION 

class HypertensionInput(BaseModel):
    Age:                  int   = Field(default=0,   ge=0, le=13, description="Age category")
    Sex:                  int   = Field(default=0,   ge=0, le=1,  description="0 = Female, 1 = Male")
    HighChol:             int   = Field(default=0,   ge=0, le=1,  description="0 = No High Cholesterol, 1 = High Cholesterol")
    CholCheck:            int   = Field(default=0,   ge=0, le=1,  description="0 = No Cholesterol Check in 5 Years, 1 = Yes")
    BMI:                  float = Field(default=0.0, ge=0, le=99, description="Body Mass Index")
    Smoker:               int   = Field(default=0,   ge=0, le=1,  description="Smoked 100+ cigarettes in life (0 = No, 1 = Yes)")
    HeartDiseaseorAttack: int   = Field(default=0,   ge=0, le=1,  description="Coronary heart disease or MI (0 = No, 1 = Yes)")
    PhysActivity:         int   = Field(default=0,   ge=0, le=1,  description="Physical activity in past 30 days (0 = No, 1 = Yes)")
    Fruits:               int   = Field(default=0,   ge=0, le=1,  description="Consume Fruit 1+ times/day (0 = No, 1 = Yes)")
    Veggies:              int   = Field(default=0,   ge=0, le=1,  description="Consume Veggies 1+ times/day (0 = No, 1 = Yes)")
    HvyAlcoholConsump:    int   = Field(default=0,   ge=0, le=1,  description="Heavy drinker (0 = No, 1 = Yes)")
    GenHlth:              int   = Field(default=0,   ge=0, le=5,  description="General health (1 = Excellent -> 5 = Poor)")
    MentHlth:             int   = Field(default=0,   ge=0, le=30, description="Days of poor mental health in past 30 days")
    PhysHlth:             int   = Field(default=0,   ge=0, le=30, description="Physical illness days in past 30 days")
    DiffWalk:             int   = Field(default=0,   ge=0, le=1,  description="Difficulty walking or climbing stairs (0 = No, 1 = Yes)")
    Stroke:               int   = Field(default=0,   ge=0, le=1,  description="Ever had a stroke (0 = No, 1 = Yes)")
    Diabetes:             int   = Field(default=0,   ge=0, le=2,  description="0 = No, 1 = Pre-diabetes, 2 = Diabetes")

HYPERTENSION_FEATURE_ORDER = [
    "Age", "Sex", "HighChol", "CholCheck", "BMI", "Smoker",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "GenHlth", "MentHlth", "PhysHlth",
    "DiffWalk", "Stroke", "Diabetes"
]

def preprocess_hypertension_input(data: HypertensionInput) -> np.ndarray:
    values = [getattr(data, feat) for feat in HYPERTENSION_FEATURE_ORDER]
    return np.array(values, dtype=np.float32).reshape(1, -1)

def load_hypertension_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "HighBP" in df.columns:
        df = df.rename(columns={"HighBP": "target"})
    logger.info(f"Loaded hypertension dataset: {len(df)} rows")
    return df


#  HEPATITIS B 

class HepatitisBInput(BaseModel):
    age:             float = Field(default=0.0, ge=0, le=120, description="Age in years")
    sex:             int   = Field(default=0,   ge=0, le=2,   description="1 = male, 2 = female (or 0/1 depending on encoding)")
    steroid:         int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    antivirals:      int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    fatigue:         int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    malaise:         int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    anorexia:        int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    liver_big:       int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    liver_firm:      int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    spleen_palpable: int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    spiders:         int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    ascites:         int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    varices:         int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")
    bilirubin:       float = Field(default=0.0, ge=0, le=10,  description="Total bilirubin (mg/dL)")
    alk_phosphate:   float = Field(default=0.0, ge=0, le=500, description="Alkaline phosphatase (IU/L)")
    sgot:            float = Field(default=0.0, ge=0, le=600, description="SGOT / AST liver enzyme (IU/L)")
    albumin:         float = Field(default=0.0, ge=0, le=7.0, description="Serum albumin (g/dL)")
    protime:         float = Field(default=0.0, ge=0, le=100, description="Prothrombin time (seconds)")
    histology:       int   = Field(default=0,   ge=0, le=2,   description="1 = no, 2 = yes")

HEPATITIS_B_FEATURE_ORDER = [
    "age", "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia",
    "liver_big", "liver_firm", "spleen_palpable", "spiders", "ascites",
    "varices", "bilirubin", "alk_phosphate", "sgot", "albumin",
    "protime", "histology"
]

HEPATITIS_ZERO_AS_NAN_FEATURES = ["bilirubin", "alk_phosphate", "sgot", "albumin", "protime"]

def preprocess_hepatitis_b_input(data: HepatitisBInput) -> np.ndarray:
    values = []
    for feat in HEPATITIS_B_FEATURE_ORDER:
        val = getattr(data, feat)
        if feat in HEPATITIS_ZERO_AS_NAN_FEATURES and val == 0.0:
            values.append(np.nan)
        else:
            values.append(val)
    return np.array(values, dtype=np.float32).reshape(1, -1)

def load_hepatitis_b_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, na_values=["?", ""])
    df.columns = [c.strip().lower() for c in df.columns]
    
    target_candidates = ["class", "target", "outcome", "result"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"Could not find a target column in hepatitis CSV. Columns present: {list(df.columns)}")
    df = df.rename(columns={target_col: "target"})
    
    target_str = df["target"].astype(str).str.strip().str.lower()
    mapping = {"1": 1, "1.0": 1, "die": 1, "2": 0, "2.0": 0, "live": 0, "0": 0, "0.0": 0}
    df["target"] = target_str.map(mapping)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    
    col_rename = {"alk phosphate": "alk_phosphate", "alkphosphate": "alk_phosphate"}
    df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})
    for col in list(df.columns):
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    missing = [f for f in HEPATITIS_B_FEATURE_ORDER if f not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan
            
    logger.info(f"Loaded hepatitis B dataset: {len(df)} rows")
    return df