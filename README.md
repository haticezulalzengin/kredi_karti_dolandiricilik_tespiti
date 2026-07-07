# Kredi Kartı Dolandırıcılık Tespiti

Bu proje, kredi kartı işlem kayıtları üzerinden dolandırıcılık riski taşıyan işlemleri makine öğrenmesi yöntemleriyle tespit etmeyi amaçlar. Çalışma yalnızca teknik bir model denemesi değildir; aynı zamanda model çıktılarının işletme kararlarını nasıl destekleyebileceğini göstermeye odaklanan bir Yönetim Bilişim Sistemleri projesidir.

Projede açık erişimli Credit Card Fraud Detection veri seti kullanılmıştır. Veri seti yüksek derecede dengesizdir: işlemlerin çok büyük bölümü normal, çok küçük bölümü ise dolandırıcılık sınıfındadır. Bu nedenle model başarısı yalnızca accuracy ile değil; precision, recall, F1-score, AUPRC ve confusion matrix sonuçlarıyla birlikte değerlendirilmiştir.

## Projenin Amacı

- Kredi kartı işlem verilerinden dolandırıcılık ihtimali taşıyan kayıtları tespit etmek
- Dengesiz sınıf dağılımı problemini SMOTE yöntemiyle ele almak
- Logistic Regression ve Random Forest modellerini eğitip değerlendirmek
- Eğitilen modelleri kaydederek daha sonra yeni veri dosyaları üzerinde tahmin yapabilmek
- Model çıktısını işletme açısından riskli işlem önceliklendirme aracı olarak yorumlamak

## Kullanılan Yöntemler

Projede ilk olarak veri seti okunmuş, eksik değer ve tekrarlı kayıt kontrolleri yapılmıştır. Ardından `Amount` değişkeni standardize edilmiş, veri eğitim ve test setlerine ayrılmıştır. Sınıf dengesizliği yalnızca eğitim verisi üzerinde SMOTE yöntemiyle dengelenmiştir. Böylece test verisinin gerçek dünya dağılımını koruması sağlanmıştır.

Modelleme aşamasında iki farklı algoritma kullanılmıştır:

- **Logistic Regression:** Karşılaştırma için temel model olarak kullanılmıştır.
- **Random Forest Classifier:** Daha güçlü bir topluluk öğrenmesi yöntemi olarak değerlendirilmiştir.

## Proje Dosya Yapısı

```text
.
├── fraud_detection_thesis.py
├── artifacts/
│   ├── logistic_bundle.joblib
│   └── random_forest_bundle.joblib
└── dosyalar/
    ├── creditcard.csv
    └── creditcard_predictions.csv
```

| Dosya / Klasör | Açıklama |
| --- | --- |
| `fraud_detection_thesis.py` | Eğitim, değerlendirme ve tahmin işlemlerini içeren ana Python dosyası |
| `dosyalar/creditcard.csv` | Model eğitimi için kullanılan kredi kartı işlem veri seti |
| `dosyalar/creditcard_predictions.csv` | Tahmin sonucunda oluşan çıktı dosyası |
| `artifacts/logistic_bundle.joblib` | Kaydedilmiş Logistic Regression modeli |
| `artifacts/random_forest_bundle.joblib` | Kaydedilmiş Random Forest modeli |

## Kurulum

Projeyi çalıştırmak için Python kurulu olmalıdır. Gerekli kütüphaneler aşağıdaki komutla kurulabilir:

```bash
pip install pandas scikit-learn imbalanced-learn joblib
```

Eğer veri dosyası repoda yer almıyorsa, `creditcard.csv` dosyası `dosyalar/` klasörünün içine eklenmelidir. Veri seti Kaggle üzerindeki Credit Card Fraud Detection veri setidir.

## Modeli Eğitme ve Değerlendirme

Aşağıdaki komut veri setini okuyarak modelleri eğitir, test verisi üzerinde değerlendirir ve eğitilen modelleri `artifacts/` klasörüne kaydeder:

```bash
python fraud_detection_thesis.py --data dosyalar/creditcard.csv
```

Kod çalıştığında konsolda şu bilgiler raporlanır:

- Veri setinin boyutu
- Eksik değer sayısı
- Tekrarlı kayıt sayısı
- Sınıf dağılımı
- SMOTE öncesi ve sonrası eğitim sınıf dağılımı
- Logistic Regression performans metrikleri
- Random Forest performans metrikleri
- Confusion matrix sonuçları

## Yeni Veri Üzerinde Tahmin Yapma

Eğitilmiş modeli kullanarak yeni bir CSV dosyası üzerinde tahmin yapılabilir:

```bash
python fraud_detection_thesis.py --predict dosyalar/creditcard.csv --model random_forest --output dosyalar/creditcard_predictions.csv
```

Bu işlem sonunda çıktı dosyasına iki yeni alan eklenir:

| Alan | Açıklama |
| --- | --- |
| `predicted_class` | İşlemin model tarafından normal veya fraud olarak sınıflandırılması |
| `fraud_probability` | İşlemin fraud olma olasılığı |

## Performans Değerlendirme

Bu projede özellikle dengesiz veri problemi bulunduğu için accuracy tek başına yeterli kabul edilmemiştir. Model çıktıları aşağıdaki metriklerle birlikte yorumlanmıştır:

- **Accuracy:** Genel doğru tahmin oranı
- **Precision:** Fraud olarak işaretlenen işlemlerin ne kadarının gerçekten fraud olduğunu gösterir
- **Recall:** Gerçek fraud işlemlerinin ne kadarının yakalandığını gösterir
- **F1-score:** Precision ve recall dengesini özetler
- **AUPRC:** Dengesiz veri setlerinde fraud sınıfına odaklı performansı daha anlamlı gösterir
- **Confusion Matrix:** Modelin hangi tür hataları yaptığını açıkça gösterir

Random Forest modeli, bu çalışmada fraud işlemlerini belirlemede daha güçlü sonuçlar üretmiştir. Ancak gerçek işletme senaryolarında modelin yalnızca otomatik karar veren bir yapı olarak değil, riskli işlemleri önceliklendiren bir karar destek aracı olarak kullanılması daha uygundur.

## YBS Açısından Değerlendirme

Bu proje Yönetim Bilişim Sistemleri bakış açısıyla ele alınmıştır. Bu nedenle projenin temel değeri yalnızca kodun çalışması değil, koddan elde edilen sonuçların işletme kararlarına dönüştürülebilmesidir.

Modelin ürettiği `fraud_probability` alanı, işletmelerde riskli işlemleri sıralamak ve inceleme ekiplerinin önceliklerini belirlemek için kullanılabilir. Böylece veri, doğrudan karar destek sürecine katkı sağlayan anlamlı bir bilgiye dönüşmektedir.

## Notlar

- SMOTE yalnızca eğitim verisine uygulanmıştır; test verisi gerçek dağılımını korumuştur.
- Model dosyaları `artifacts/` klasörüne kaydedilir.
- Tahmin modunun çalışabilmesi için önce eğitim modunda model dosyalarının oluşturulmuş olması gerekir.
- Veri seti büyük boyutlu olduğu için GitHub'a yüklenmemiş olabilir. Bu durumda `creditcard.csv` dosyası manuel olarak `dosyalar/` klasörüne eklenmelidir.

## Geliştirici

**Hatice Zülal Zengin**  
Yönetim Bilişim Sistemleri
