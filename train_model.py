import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# LOAD DATASET


print("Loading dataset...")

df = pd.read_csv("placement_dataset.csv")

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset columns:")
print(df.columns.tolist())



# FEATURES


features = [
    "college_name",
    "branch",
    "semester",
    "cgpa",
    "aptitude_score",
    "technical_skill_score",
    "communication_skill_score",
    "coding_score",
    "internship_experience",
    "projects",
    "backlogs",
    "certifications"
]


# Check columns
missing_columns = [
    column for column in features
    if column not in df.columns
]

if missing_columns:

    print("\nERROR!")
    print("Missing columns:")

    for column in missing_columns:
        print("-", column)

    raise SystemExit()

# X AND Y


X = df[features]

y = df["placement_status"]



# CATEGORICAL FEATURES


categorical_features = [
    "college_name",
    "branch"
]



# NUMERICAL FEATURES


numerical_features = [
    "semester",
    "cgpa",
    "aptitude_score",
    "technical_skill_score",
    "communication_skill_score",
    "coding_score",
    "internship_experience",
    "projects",
    "backlogs",
    "certifications"
]



# PREPROCESSING


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)



# MODEL


classifier = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced"
)



# PIPELINE


pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )
    ]
)



# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)



# TRAIN


print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)



# TEST


predictions = pipeline.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n================================")
print("MODEL TRAINING COMPLETE")
print("================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)



# SAVE MODEL


with open(
    "placement_model.pkl",
    "wb"
) as file:

    pickle.dump(
        pipeline,
        file
    )


print("\nModel saved successfully!")
print("File: placement_model.pkl")