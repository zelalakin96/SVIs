# Google Street View Görüntü İndirme Otomasyonu

Bu proje, bir Excel dosyasındaki koordinatlara (enlem ve boylam) ait Google Street View görüntülerini otomatik olarak indirir ve zip dosyası olarak sunar.

## Kullanım Adımları

1. **API Anahtarınızı Girin:**
   - `svi.py` dosyasındaki `GOOGLE_STREET_VIEW_API_KEY` değişkenine kendi Google Street View API anahtarınızı girin.
2. **Excel Dosyasını Yükleyin:**
   - Programı çalıştırdığınızda, koordinat verilerini içeren Excel dosyasını yüklemeniz istenir. (B sütunu: boylam/x, C sütunu: enlem/y, ayrıca STUDENTID sütunu olmalı)
3. **Otomatik İşlem:**
   - Program, Excel dosyasını okur, tekrar eden koordinatları filtreler.
   - Her benzersiz koordinat için 0°, 90°, 180°, 270° başlıklarında sokak görünümü görsellerini indirir.
   - Tüm görselleri `street_view_images` klasörüne kaydeder.
   - Son olarak, bu klasörü zipleyip indirmenizi sağlar.

## Kullanılan Komutlar

- `git status`: Çalışma dizinindeki ve geçici alandaki (staging area) dosyaların durumunu gösterir.
- `git add .`: Bulunulan dizindeki tüm değişiklikleri geçici alana ekler.
- `git commit -m "mesaj"`: Geçici alandaki değişiklikleri, verilen mesaj ile birlikte kalıcı olarak kaydeder.
- `git push`: Yerel depodaki commit'leri uzak depoya gönderir.

## Notlar
- Excel dosyanızda `STUDENTID`, `x` (boylam), `y` (enlem) sütunları bulunmalıdır.
- Google API anahtarınızın Street View Static API için etkinleştirildiğinden emin olun.
- Kod Google Colab ortamında çalışacak şekilde tasarlanmıştır.