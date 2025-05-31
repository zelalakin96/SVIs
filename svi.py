import pandas as pd
import requests
import io
import os
from google.colab import files
from PIL import Image  # For image checking using Pillow library
import shutil

# **1. API Anahtarınızı Girin**
GOOGLE_STREET_VIEW_API_KEY = "AIzaSyA0P9qlaExBaOLYmeIu3OoxNDfdXx9C978"  # Kendi API anahtarınızı buraya yapıştırın!

# **2. Excel Dosyalarını Yükleme**
print("Lütfen koordinat verilerini içeren Excel dosyalarını yükleyin (B sütunu boylam, C sütunu enlem olmalı).")
uploaded_files = files.upload()

# **3. Excel Dosyaları İçin İşleme Fonksiyonu**
def process_excel_file(excel_file_name, file_content):
    # **3. Excel Dosyasını Okuma**
    try:
        df = pd.read_excel(io.BytesIO(file_content), decimal=',')
        print(f"Excel dosyası '{excel_file_name}' başarıyla okundu.")
    except Exception as e:
        print(f"Excel dosyası okuma hatası: {e}")
        return

    # **4. Tekrar Eden Koordinatları Filtreleme**
    print("Tekrar eden koordinatlar kontrol ediliyor...")
    initial_coordinate_count = len(df)
    df.drop_duplicates(subset=['x', 'y'], keep='first', inplace=True)  # 'x' ve 'y' sütunlarına göre tekrar edenleri sil
    unique_coordinate_count = len(df)
    if unique_coordinate_count < initial_coordinate_count:
        print(f"Tekrar eden {initial_coordinate_count - unique_coordinate_count} koordinat bulundu ve kaldırıldı. {unique_coordinate_count} benzersiz koordinat işlenecek.")
    else:
        print("Tekrar eden koordinat bulunamadı.")

    # **5. Görüntüleri Kaydetmek İçin Klasör Oluşturma**
    output_folder = "street_view_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' klasörü oluşturuldu.")

    # **6. Sokak Görünümü İndirme Fonksiyonu**
    def indir_sokak_gorunumu(student_id, lat, lon, heading, output_path):
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "size": "640x640",  # Görüntü boyutu
            "location": f"{lat},{lon}",
            "heading": heading,
            "fov": "90",
            "pitch": "0",
            "key": GOOGLE_STREET_VIEW_API_KEY,
        }

        try:
            response = requests.get(url, params=params, stream=True)
            response.raise_for_status()  # Hata durumunda HTTPError yükseltir (4xx veya 5xx hataları)

            if response.headers['content-type'] == 'image/jpeg':  # Görüntü içeriği kontrolü
                with open(output_path, "wb") as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)
                print(f" -> Başlık: {heading}° - Görüntü kaydedildi: {output_path}")  # Daha detaylı çıktı
                return True
            else:
                print(f" -> Başlık: {heading}° - Görüntü alınamadı (Görüntü değil): {response.headers['content-type']}")  # Daha detaylı çıktı
                return False

        except requests.exceptions.HTTPError as errh:
            print(f" -> Başlık: {heading}° - HTTP Hatası: {errh}")  # Daha detaylı çıktı
            return False
        except requests.exceptions.ConnectionError as errc:
            print(f" -> Başlık: {heading}° - Bağlantı Hatası: {errc}")  # Daha detaylı çıktı
            return False
        except requests.exceptions.Timeout as errt:
            print(f" -> Başlık: {heading}° - Zaman Aşımı Hatası: {errt}")  # Daha detaylı çıktı
            return False
        except requests.exceptions.RequestException as err:
            print(f" -> Başlık: {heading}° - İstek Hatası: {err}")  # Daha detaylı çıktı
            return False

    # **7. Koordinatları İşleme ve Görüntüleri İndirme**
    print(f"\nSokak görünümleri '{excel_file_name}' dosyası için indiriliyor...")
    for index, row in df.iterrows():
        try:
            student_id = row['STUDENTID']
            lat = row['y']  # veya df.iloc[index, 2] sütun indeksine göre (C sütunu - ENLEM)
            lon = row['x']  # veya df.iloc[index, 1] sütun indeksine göre (B sütunu - BOYLAM)

            if pd.isna(lat) or pd.isna(lon):  # Boş koordinat kontrolü
                print(f"Satır {index+2}: Geçersiz koordinatlar (boş değer). Atlanıyor.")  # Excel satır numarası için +2
                continue

            try:
                lat = float(lat)
                lon = float(lon)
            except ValueError:
                print(f"Satır {index+2}: Geçersiz koordinatlar (sayısal değil). Atlanıyor.")
                continue

            print(f"Satır {index+2}: Koordinat {index+1} - StudentID: {student_id}, Enlem: {lat}, Boylam: {lon}")  # Daha detaylı çıktı

            for heading in [0, 90, 180, 270]:
                dosya_adi = f"street_view_{student_id}_lat{lat}_lon{lon}_heading{heading}.jpg"
                output_path = os.path.join(output_folder, dosya_adi)
                indir_sokak_gorunumu(student_id, lat, lon, heading, output_path)

        except Exception as genel_hata:
            print(f"Satır {index+2} işlenirken genel hata oluştu: {genel_hata}")

    print(f"\nSokak görünümleri '{excel_file_name}' dosyası için indirme tamamlandı.")

    # **8. İndirilen Görüntüleri Zip Dosyası Olarak İndirme**
    zip_file_name = f"{output_folder}.zip"
    shutil.make_archive(output_folder, 'zip', output_folder)  # Klasörü zip yap
    print(f"\nİndirilen görüntüler '{zip_file_name}' olarak zip dosyasına kaydedildi.")
    files.download(zip_file_name)  # Zip dosyasını indir
