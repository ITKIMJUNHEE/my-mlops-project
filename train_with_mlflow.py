import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
from mlflow.tracking import MlflowClient

# ── MLflow 연결 설정 ──────────────────────────────────────────
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

if "MLFLOW_TRACKING_USERNAME" in os.environ:
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

experiment_name = "iris_classification"
mlflow.set_experiment(experiment_name)

# ── 1. 데이터 로드 및 경로 확인 (수정됨) ──────────────────────────
# 로봇이 파일이 어디 있는지 못 찾을 때를 대비해 경로 후보를 여러 개 체크합니다.
data_paths = ["data/iris_data.csv", "src/main/data/iris_data.csv", "iris_data.csv"]
df = None

print(f"🔍 현재 위치: {os.getcwd()}")
for path in data_paths:
    if os.path.exists(path):
        print(f"✅ 데이터를 찾았습니다: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    print("❌ 데이터를 찾을 수 없습니다. 아래 목록을 확인하세요:")
    # 디버깅을 위해 현재 폴더 구조를 출력합니다.
    for root, dirs, files in os.walk("."):
        for file in files:
            if "iris" in file:
                print(f"찾은 파일: {os.path.join(root, file)}")
    exit(1)

# 전처리: 'target' 컬럼을 유지하면서 수치형 전처리 진행
try:
    # 수치형 컬럼과 정답(target) 컬럼만 남깁니다.
    cols_to_keep = df.select_dtypes(include=['number']).columns.tolist()
    if 'target' not in cols_to_keep and 'target' in df.columns:
        cols_to_keep.append('target')
    
    df = df[cols_to_keep].dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ 데이터 로드 및 분리 완료")
except KeyError:
    print(f"❌ 'target' 컬럼이 데이터에 없습니다. 실제 컬럼명: {df.columns.tolist()}")
    exit(1)

# ── 2. 실험 파라미터 목록 정의 ────────────────────────────────
run_results = []
param_list = [
    {"n_estimators": 60,  "max_depth": 3},
    {"n_estimators": 120, "max_depth": 4},
    {"n_estimators": 250, "max_depth": 5},
]

# ── 3. 파라미터별 실험 실행 ───────────────────────────────────
for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}"
    with mlflow.start_run(run_name=run_name):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**params, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        model_info = mlflow.sklearn.log_model(pipe, artifact_path="model")

        run_results.append({
            "run_name": run_name,
            "accuracy": acc,
            "model_uri": model_info.model_uri
        })
        print(f"  {run_name}: {acc:.4f}")

# ── 4. 최고 모델 선택 및 등록 ─────────────────────────────────
best = max(run_results, key=lambda x: x["accuracy"])
print(f"🏆 최고 모델: {best['run_name']} ({best['accuracy']:.4f})")

registered = mlflow.register_model(model_uri=best["model_uri"], name="iris_classifier")
client = MlflowClient()
client.set_registered_model_alias(name="iris_classifier", alias="production", version=registered.version)
print(f"🚀 Production 등록 완료 (Version {registered.version})")