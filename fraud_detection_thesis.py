import argparse
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def print_section(title: str) -> None:
    """Console output'u daha okunakli hale getirmek icin baslik yazdirir."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def evaluate_model(model_name: str, y_true: pd.Series, y_pred, y_score) -> None:
    """Model performans metriklerini akademik raporlama duzeninde yazdirir."""
    print_section(f"{model_name} - Test Performansi")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"AUPRC    : {average_precision_score(y_true, y_score):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


def resolve_data_path(cli_path: str | None = None) -> Path:
    """Veri dosyasini once argumandan, sonra script klasorunden bulmaya calisir."""
    script_dir = Path(__file__).resolve().parent
    candidate_paths = []

    if cli_path:
        candidate_paths.append(Path(cli_path).expanduser())

    candidate_paths.extend(
        [
            script_dir / "creditcard.csv",
            script_dir / "dosyalar" / "creditcard.csv",
            Path.cwd() / "creditcard.csv",
            Path.cwd() / "dosyalar" / "creditcard.csv",
        ]
    )

    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path.resolve()

    searched_locations = "\n".join(f"- {path.resolve()}" for path in candidate_paths)
    raise FileNotFoundError(
        "creditcard.csv dosyasi bulunamadi.\n"
        "Kontrol edilen konumlar:\n"
        f"{searched_locations}\n\n"
        "Cozum: CSV dosyasini bu konumlardan birine ekleyin veya "
        "`python fraud_detection_thesis.py --data DOSYA_YOLU` ile acik yol verin."
    )


def resolve_optional_data_path(cli_path: str | None) -> Path:
    """Tahmin modunda kullanilacak test dosyasinin yolunu dogrular."""
    if not cli_path:
        raise ValueError(
            "Tahmin modu icin bir dosya yolu vermelisiniz. "
            "Ornek: python fraud_detection_thesis.py --predict test.csv"
        )

    path = Path(cli_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Tahmin dosyasi bulunamadi: {path}")
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit card fraud detection tez calismasi."
    )
    parser.add_argument(
        "--data",
        dest="data_path",
        help="creditcard.csv dosyasinin tam veya goreli yolu",
    )
    parser.add_argument(
        "--predict",
        dest="predict_path",
        help="Tahmin yapmak icin kullanilacak yeni test CSV dosyasinin yolu",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest"],
        default="random_forest",
        help="Tahmin modunda yuklenecek egitilmis model",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Tahmin sonuclarinin kaydedilecegi CSV dosyasi yolu",
    )
    return parser.parse_args()


def validate_dataset_columns(df: pd.DataFrame) -> None:
    """Veri setinin beklenen kredi karti dolandiriciligi semasina uydugunu kontrol eder."""
    required_columns = {"Time", "Amount", "Class"}
    required_columns.update({f"V{i}" for i in range(1, 29)})

    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Veri setinde beklenen sutunlar eksik: "
            + ", ".join(missing_columns)
        )


def build_feature_matrix(
    df: pd.DataFrame, scaler: StandardScaler, fit_scaler: bool
) -> tuple[pd.DataFrame, pd.Series]:
    """Model icin gerekli ozellik matrisini ve hedef degiskeni hazirlar."""
    working_df = df.copy()

    if fit_scaler:
        working_df["normAmount"] = scaler.fit_transform(working_df[["Amount"]])
    else:
        working_df["normAmount"] = scaler.transform(working_df[["Amount"]])

    working_df = working_df.drop(columns=["Time", "Amount"])
    X = working_df.drop(columns=["Class"])
    y = working_df["Class"]
    return X, y


def build_prediction_matrix(
    df: pd.DataFrame, scaler: StandardScaler, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Yeni test verisini egitim sirasindaki ozellik yapisina donusturur."""
    working_df = df.copy()
    has_labels = "Class" in working_df.columns

    working_df["normAmount"] = scaler.transform(working_df[["Amount"]])
    working_df = working_df.drop(columns=["Time", "Amount"])

    y = working_df["Class"] if has_labels else None
    if has_labels:
        working_df = working_df.drop(columns=["Class"])

    missing_features = sorted(set(feature_columns).difference(working_df.columns))
    extra_features = sorted(set(working_df.columns).difference(feature_columns))

    if missing_features:
        raise ValueError(
            "Tahmin verisinde eksik ozellikler var: " + ", ".join(missing_features)
        )
    if extra_features:
        raise ValueError(
            "Tahmin verisinde beklenmeyen ek ozellikler var: " + ", ".join(extra_features)
        )

    X = working_df[feature_columns]
    return X, y


def save_artifact(model_name: str, model, scaler: StandardScaler, feature_columns: list[str]) -> Path:
    """Model, scaler ve ozellik bilgisini tekrar kullanmak uzere diske kaydeder."""
    ARTIFACT_DIR.mkdir(exist_ok=True)
    artifact_path = ARTIFACT_DIR / f"{model_name}_bundle.joblib"
    joblib.dump(
        {
            "model_name": model_name,
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
        },
        artifact_path,
    )
    return artifact_path


def load_artifact(model_name: str) -> dict:
    """Tahmin modunda kaydedilmis modeli yukler."""
    artifact_path = ARTIFACT_DIR / f"{model_name}_bundle.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Egitilmis model bulunamadi: {artifact_path}\n"
            "Once egitim modunda scripti calistirarak modeli olusturun."
        )
    return joblib.load(artifact_path)


def train_and_evaluate(data_path: Path) -> None:
    """Egitim, degerlendirme ve model kaydetme akisini uctan uca calistirir."""
    scaler = StandardScaler()

    print_section("1. Veri Yukleme ve Genel Kontrol")

    # Veri setini Pandas ile okuyoruz.
    df = pd.read_csv(data_path)
    validate_dataset_columns(df)
    print(f"Kullanilan veri dosyasi: {data_path}")
    print(f"Veri seti boyutu (ilk durum): {df.shape}")

    # Eksik deger kontrolu.
    missing_values = df.isnull().sum().sum()
    print(f"Toplam eksik deger sayisi: {missing_values}")

    # Tekrarlayan kayitlari kontrol edip temizliyoruz.
    duplicate_count = df.duplicated().sum()
    print(f"Tekrarlayan kayit sayisi: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Tekrarlayan kayitlar silindi. Yeni boyut: {df.shape}")
    else:
        print("Tekrarlayan kayit bulunmadi.")

    # Sinif dagilimini hem adet hem oran olarak gosteriyoruz.
    class_counts = df["Class"].value_counts().sort_index()
    class_ratios = df["Class"].value_counts(normalize=True).sort_index() * 100

    print("\nSinif Dagilimi:")
    print(f"0 (Normal): {class_counts.get(0, 0)} kayit ({class_ratios.get(0, 0):.4f}%)")
    print(f"1 (Fraud) : {class_counts.get(1, 0)} kayit ({class_ratios.get(1, 0):.4f}%)")

    print_section("2. Veri On Isleme")

    # Olcekleyici yalnizca egitim setinde fit edilerek veri sizintisi onlenir.
    X_raw = df.drop(columns=["Class"])
    y = df["Class"]

    # Veri dengesiz oldugu icin stratified train-test split kullaniyoruz.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    train_df = X_train_raw.copy()
    train_df["Class"] = y_train.values
    test_df = X_test_raw.copy()
    test_df["Class"] = y_test.values

    X_train, y_train = build_feature_matrix(train_df, scaler, fit_scaler=True)
    X_test, y_test = build_feature_matrix(test_df, scaler, fit_scaler=False)

    print(f"Egitim veri boyutu: {X_train.shape}")
    print(f"Test veri boyutu  : {X_test.shape}")

    print_section("3. SMOTE ile Veri Dengeleme")

    # Sadece egitim verisine SMOTE uygulayarak azinlik sinifini dengeliyoruz.
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    original_train_distribution = y_train.value_counts().sort_index()
    resampled_train_distribution = y_train_resampled.value_counts().sort_index()

    print("SMOTE oncesi egitim sinif dagilimi:")
    print(original_train_distribution)

    print("\nSMOTE sonrasi egitim sinif dagilimi:")
    print(resampled_train_distribution)

    print_section("4. Modelleme")

    # Temel model olarak lojistik regresyon egitiyoruz.
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train_resampled, y_train_resampled)
    y_pred_logistic = logistic_model.predict(X_test)
    y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]

    # Daha guclu bir ensemble yaklasim olarak Random Forest egitiyoruz.
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    random_forest_model.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = random_forest_model.predict(X_test)
    y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]

    logistic_artifact = save_artifact(
        "logistic",
        logistic_model,
        scaler,
        X_train.columns.tolist(),
    )
    random_forest_artifact = save_artifact(
        "random_forest",
        random_forest_model,
        scaler,
        X_train.columns.tolist(),
    )

    print("Her iki model de basariyla egitildi ve test verisi uzerinde tahmin yapti.")
    print(f"Kaydedilen Logistic Regression modeli: {logistic_artifact}")
    print(f"Kaydedilen Random Forest modeli      : {random_forest_artifact}")

    print_section("5. Akademik Degerlendirme")

    # Her iki modeli ayni metriklerle degerlendirip karsilastiriyoruz.
    evaluate_model("Logistic Regression", y_test, y_pred_logistic, y_score_logistic)
    evaluate_model("Random Forest Classifier", y_test, y_pred_rf, y_score_rf)


def predict_new_file(predict_path: Path, model_name: str, output_path: str | None) -> None:
    """Kaydedilmis modeli kullanarak yeni bir test dosyasi icin tahmin uretir."""
    print_section("Tahmin Modu")
    artifact = load_artifact(model_name)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    df = pd.read_csv(predict_path)
    validate_dataset_columns(df)

    X_new, y_true = build_prediction_matrix(df, scaler, feature_columns)
    y_pred = model.predict(X_new)
    y_score = model.predict_proba(X_new)[:, 1]

    results_df = df.copy()
    results_df["predicted_class"] = y_pred
    results_df["fraud_probability"] = y_score

    if output_path:
        output_file = Path(output_path).expanduser().resolve()
    else:
        output_file = predict_path.with_name(f"{predict_path.stem}_predictions.csv")

    results_df.to_csv(output_file, index=False)

    print(f"Kullanilan model: {model_name}")
    print(f"Tahmin yapilan dosya: {predict_path}")
    print(f"Sonuclar kaydedildi: {output_file}")
    print(f"Toplam kayit sayisi: {len(results_df)}")
    print(f"Tahmin edilen fraud sayisi: {int(results_df['predicted_class'].sum())}")

    if y_true is not None:
        evaluate_model(model_name.replace("_", " ").title(), y_true, y_pred, y_score)


def main() -> None:
    args = parse_args()
    if args.predict_path:
        predict_path = resolve_optional_data_path(args.predict_path)
        predict_new_file(predict_path, args.model, args.output_path)
    else:
        data_path = resolve_data_path(args.data_path)
        train_and_evaluate(data_path)


if __name__ == "__main__":
    main()
