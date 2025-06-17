import pandas as pd
import requests
import io
import os
from google.colab import files
from PIL import Image  # For image checking using Pillow library
import shutil

# Session information
CURRENT_TIME = "2025-06-17 12:49:22"
CURRENT_USER = "zelalakin96"

# **1. API Key**
GOOGLE_STREET_VIEW_API_KEY = "API_key"

# **2. Upload Excel Files**
print("Please upload Excel files containing coordinate data (columns: code, x, y, point_id)")
uploaded_files = files.upload()

# **3. Processing Function for Excel Files**
def process_excel_file(excel_file_name, file_content):
    # **3. Reading Excel File**
    try:
        df = pd.read_excel(io.BytesIO(file_content), decimal=',')
        print(f"Excel file '{excel_file_name}' successfully read.")
    except Exception as e:
        print(f"Excel file reading error: {e}")
        return

    # **4. Create Folder for Images**
    output_folder = "street_view_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' folder created.")

    # **5. Street View Download Function**
    def download_street_view(code, point_id, lat, lon, heading, output_path):
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "size": "640x640",
            "location": f"{lat},{lon}",
            "heading": heading,
            "fov": "90",
            "pitch": "0",
            "key": GOOGLE_STREET_VIEW_API_KEY,
        }

        try:
            response = requests.get(url, params=params, stream=True)
            response.raise_for_status()

            if response.headers['content-type'] == 'image/jpeg':
                with open(output_path, "wb") as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)
                print(f" -> Heading: {heading}° - Image saved: {output_path}")
                return True
            else:
                print(f" -> Heading: {heading}° - Failed to get image (Not an image): {response.headers['content-type']}")
                return False

        except requests.exceptions.HTTPError as errh:
            print(f" -> Heading: {heading}° - HTTP Error: {errh}")
            return False
        except requests.exceptions.ConnectionError as errc:
            print(f" -> Heading: {heading}° - Connection Error: {errc}")
            return False
        except requests.exceptions.Timeout as errt:
            print(f" -> Heading: {heading}° - Timeout Error: {errt}")
            return False
        except requests.exceptions.RequestException as err:
            print(f" -> Heading: {heading}° - Request Error: {err}")
            return False

    # **6. Process Coordinates and Download Images**
    print(f"\nDownloading street views for file '{excel_file_name}'...")
    for index, row in df.iterrows():
        try:
            code = row['code']        # S1, S2, etc.
            point_id = row['point_id']
            lat = row['y']  # latitude
            lon = row['x']  # longitude

            if pd.isna(lat) or pd.isna(lon):
                print(f"Row {index+2}: Invalid coordinates (empty value). Skipping.")
                continue

            try:
                lat = float(str(lat).replace(',', '.'))
                lon = float(str(lon).replace(',', '.'))
            except ValueError:
                print(f"Row {index+2}: Invalid coordinates (not numeric). Skipping.")
                continue

            print(f"Row {index+2}: Coordinate {index+1} - Code: {code}, Point ID: {point_id}, Lat: {lat}, Lon: {lon}")

            for heading in [0, 90, 180, 270]:
                # New file naming format: S1_p0001_0.jpg
                formatted_point_id = str(point_id).zfill(4)
                file_name = f"{code}_p{formatted_point_id}_{heading}.jpg"
                output_path = os.path.join(output_folder, file_name)
                download_street_view(code, point_id, lat, lon, heading, output_path)

        except Exception as general_error:
            print(f"General error occurred while processing row {index+2}: {general_error}")

    print(f"\nStreet view download completed for file '{excel_file_name}'.")

    # **7. Download Images as Zip File**
    zip_file_name = f"{output_folder}.zip"
    shutil.make_archive(output_folder, 'zip', output_folder)
    print(f"\nDownloaded images have been saved as '{zip_file_name}'.")
    files.download(zip_file_name)

# Process each uploaded file
for filename, content in uploaded_files.items():
    print(f"\nProcessing file: {filename}")
    process_excel_file(filename, content)
