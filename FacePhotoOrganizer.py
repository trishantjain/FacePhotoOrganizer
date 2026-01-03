import os
import shutil
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
import io
import cv2
import face_recognition
from sklearn.cluster import DBSCAN
import numpy as np
from pathlib import Path

#define Google Drive API constantst
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = 'credentials.json'
DOWNLOAD_DIR = 'downloaded_photos'
REFERENCE_FACES_DIR = 'reference_faces'  # Folder containing photos of people you want to find
FACE_MATCH_TOLERANCE = 0.4  # Lower = stricter matching (0.4-0.6 recommended)
MAX_IMAGE_WIDTH = 800  # Resize large images for faster processing
DEBUG_MODE = True  # Set to True to see face distance values
MAX_RETRIES = 3  # Number of retry attempts for failed downloads
PARALLEL_WORKERS = 6  # Number of folders to process in parallel

# Set this to your shared folder ID (from the URL: drive.google.com/drive/folders/THIS_IS_THE_ID)
SHARED_FOLDER_ID = "ID OF DRIVE FOLDER"  # e.g., "1ABC123xyz..." - Set to None to scan entire Drive


def authenticate_google_drive():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('drive', 'v3', credentials=creds)
    return service


def load_reference_faces(reference_folder):
    """Load face encodings from reference photos."""
    reference_encodings = {}
    
    # Generating the folder if it doesn't exist
    if not os.path.exists(reference_folder):
        os.makedirs(reference_folder)
        print(f"Created '{reference_folder}' folder. Add photos of people you want to find.")
        return reference_encodings
    
    # Loading file from folder
    for filename in os.listdir(reference_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(reference_folder, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image) # Finding face details which returns array
            
            if encodings:
                # Use filename (without extension) as person name
                person_name = os.path.splitext(filename)[0]
                reference_encodings[person_name] = encodings[0]
                print(f"Loaded reference face: {person_name}")
            else:
                print(f"Warning: No face found in {filename}")
    
    return reference_encodings


def resize_image_if_needed(image, max_width=800):
    """Resize image if it's too large, for faster processing."""
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    return image


def check_photo_for_matches(image_data, reference_encodings, tolerance=0.4):
    """Check if image contains any of the reference faces."""
    # Load image from bytes
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        print("[ERROR] Could not decode image")
        return []
    
    # Resize for faster processing
    image = resize_image_if_needed(image, MAX_IMAGE_WIDTH)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find faces in the image
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    
    if not face_locations:
        if DEBUG_MODE:
            print(f"[No faces detected] ", end="")
        return []
    
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    matched_people = []
    
    for face_encoding in face_encodings:
        best_match = None
        best_distance = float('inf')
        
        for person_name, ref_encoding in reference_encodings.items():
            # Calculate face distance (lower = more similar)
            face_distance = face_recognition.face_distance([ref_encoding], face_encoding)[0]
            
            if DEBUG_MODE:
                print(f"\n    {person_name}: {face_distance:.3f}", end="")
            
            # Check if this is a match AND the best match so far
            if face_distance < tolerance and face_distance < best_distance:
                best_distance = face_distance
                best_match = person_name
        
        if best_match:
            matched_people.append(best_match)
    
    if DEBUG_MODE:
        print()  # New line after distances
    
    return matched_people


def get_subfolders(service, parent_folder_id):
    """Get all subfolders within a parent folder."""
    query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(
        q=query,
        pageSize=100,
        fields="files(id, name)"
    ).execute()
    return results.get('files', [])


def get_photos_in_folder(service, folder_id):
    """Get all photos within a specific folder (including nested)."""
    all_photos = []
    
    # Get photos directly in this folder
    query = f"'{folder_id}' in parents and (mimeType='image/jpeg' or mimeType='image/png') and trashed=false"
    page_token = None
    
    while True:
        results = service.files().list(
            q=query,
            pageSize=500,
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()
        all_photos.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        if not page_token:
            break
    
    # Also get photos from subfolders recursively
    subfolders = get_subfolders(service, folder_id)
    for subfolder in subfolders:
        all_photos.extend(get_photos_in_folder(service, subfolder['id']))
    
    return all_photos


def process_folder(folder_info, reference_encodings, download_folder):
    """Process a single folder - designed to run in parallel."""
    folder_id, folder_name = folder_info['id'], folder_info['name']
    print(f"\n?? Starting folder: {folder_name}")
    
    # Each thread needs its own service instance
    service = authenticate_google_drive()
    
    # Get all photos in this folder
    photos = get_photos_in_folder(service, folder_id)
    print(f"?? {folder_name}: Found {len(photos)} photos")
    
    downloaded_paths = []
    skipped_files = []
    
    for idx, item in enumerate(photos, 1):
        print(f"[{folder_name}] [{idx}/{len(photos)}] {item['name']}...", end=" ")
        
        # Download with retry logic
        image_data = None
        for attempt in range(MAX_RETRIES):
            try:
                request = service.files().get_media(fileId=item['id'])
                file_buffer = io.BytesIO()
                downloader = MediaIoBaseDownload(file_buffer, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                image_data = file_buffer.getvalue()
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    print(f"? Failed: {str(e)[:30]}")
                    skipped_files.append(item['name'])
        
        if image_data is None:
            continue
        
        try:
            matched_people = check_photo_for_matches(image_data, reference_encodings, FACE_MATCH_TOLERANCE)
        except Exception as e:
            print(f"? Error: {str(e)[:30]}")
            skipped_files.append(item['name'])
            continue
        
        if matched_people:
            for person_name in set(matched_people):
                person_folder = os.path.join(download_folder, person_name)
                os.makedirs(person_folder, exist_ok=True)
                filepath = os.path.join(person_folder, item['name'])
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                downloaded_paths.append(filepath)
            print(f"? {', '.join(set(matched_people))}")
        else:
            print("?")
    
    print(f"\n? Folder '{folder_name}' done: {len(downloaded_paths)} matches, {len(skipped_files)} skipped")
    return {'folder': folder_name, 'downloaded': downloaded_paths, 'skipped': skipped_files}


def process_folders_parallel(service, parent_folder_id, reference_encodings, download_folder):
    """Process all subfolders of a parent folder in parallel."""
    # Get all subfolders
    subfolders = get_subfolders(service, parent_folder_id)
    
    if not subfolders:
        print(f"No subfolders found in the shared folder. Scanning the folder directly...")
        # Treat the parent folder itself as the only folder to process
        subfolders = [{'id': parent_folder_id, 'name': 'SharedFolder'}]
    
    print(f"\n?? Found {len(subfolders)} folders to process: {', '.join(f['name'] for f in subfolders)}")
    print(f"?? Processing {min(len(subfolders), PARALLEL_WORKERS)} folders in parallel...\n")
    
    all_results = []
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        # Submit all folder processing tasks
        future_to_folder = {
            executor.submit(process_folder, folder, reference_encodings, download_folder): folder
            for folder in subfolders
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"? Folder '{folder['name']}' failed: {e}")
    
    return all_results


def download_matching_photos(service, download_folder, reference_encodings):
    """Download only photos that contain reference faces."""
    results = service.files().list(
        q="mimeType='image/jpeg' or mimeType='image/png'", 
        pageSize=500,
        fields="files(id, name)"
    ).execute()
    items = results.get('files', [])

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    downloaded_paths = []
    skipped_files = []
    total_files = len(items)
    
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{total_files}] Checking {item['name']}...", end=" ")
        
        # Download to memory first to check faces - with retry logic
        image_data = None
        for attempt in range(MAX_RETRIES):
            try:
                request = service.files().get_media(fileId=item['id'])
                file_buffer = io.BytesIO()
                downloader = MediaIoBaseDownload(file_buffer, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                image_data = file_buffer.getvalue()
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"\n    ?? Download failed, retrying ({attempt + 2}/{MAX_RETRIES})...", end=" ")
                    time.sleep(2)  # Wait 2 seconds before retry
                else:
                    print(f"\n    ? Failed after {MAX_RETRIES} attempts: {str(e)[:50]}")
                    skipped_files.append(item['name'])
                    continue
        
        if image_data is None:
            continue
        
        # Check if image contains any reference faces
        try:
            matched_people = check_photo_for_matches(image_data, reference_encodings, FACE_MATCH_TOLERANCE)
        except Exception as e:
            print(f"? Error processing: {str(e)[:50]}")
            skipped_files.append(item['name'])
            continue
        
        if matched_people:
            # Save the photo to appropriate folders
            for person_name in set(matched_people):
                person_folder = os.path.join(download_folder, person_name)
                os.makedirs(person_folder, exist_ok=True)
                filepath = os.path.join(person_folder, item['name'])
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                downloaded_paths.append(filepath)
            print(f"? Match found: {', '.join(set(matched_people))}")
        else:
            print("? No match")
    
    if skipped_files:
        print(f"\n?? Skipped {len(skipped_files)} files due to errors: {', '.join(skipped_files[:5])}{'...' if len(skipped_files) > 5 else ''}")
    
    return downloaded_paths

def download_photos_from_drive(service, download_folder):
    """Original function - downloads all photos."""
    results = service.files().list(q="mimeType='image/jpeg'", pageSize=100).execute()
    items = results.get('files', [])

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    for item in items:
        request = service.files().get_media(fileId=item['id'])
        fh = io.FileIO(os.path.join(download_folder, item['name']), 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        print(f'Downloaded {item["name"]}')
    return [os.path.join(download_folder, file) for file in os.listdir(download_folder)]

def extract_face_encodings(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_encodings

def cluster_faces(encodings, eps=0.5, min_samples=2):
    clustering_model = DBSCAN(
        eps=eps, min_samples=min_samples, metric='euclidean')
    clustering_model.fit(encodings)
    return clustering_model.labels_

def organize_photos_by_faces(photo_paths):
    face_encodings = []
    photo_faces = []

    for photo_path in photo_paths:
        encodings = extract_face_encodings(photo_path)
        if encodings:
            face_encodings.extend(encodings)
            photo_faces.append((photo_path, len(encodings)))

    if face_encodings:
        clusters = cluster_faces(face_encodings)
        face_count = 0

        for i, (photo_path, num_faces) in enumerate(photo_faces):
            for j in range(num_faces):
                cluster_id = clusters[face_count]
                folder_name = f"Person_{cluster_id}" if cluster_id != - \
                    1 else "Unknown"
                target_dir = os.path.join('sorted_photos', folder_name)
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy(photo_path, target_dir)
                face_count += 1

def main():
    # Load reference faces first
    print(f"Loading reference faces from '{REFERENCE_FACES_DIR}' folder...")
    reference_encodings = load_reference_faces(REFERENCE_FACES_DIR)
    
    if not reference_encodings:
        print("\n??  No reference faces found!")
        print(f"Please add photos to the '{REFERENCE_FACES_DIR}' folder.")
        print("Name each file after the person (e.g., 'John.jpg', 'Mom.png')")
        return
    
    print(f"\nFound {len(reference_encodings)} reference face(s): {', '.join(reference_encodings.keys())}")
    
    # Authenticate and connect to Google Drive
    service = authenticate_google_drive()

    if SHARED_FOLDER_ID:
        # Process shared folder with subfolders in parallel
        print(f"\n?? Scanning shared folder: {SHARED_FOLDER_ID}")
        results = process_folders_parallel(service, SHARED_FOLDER_ID, reference_encodings, DOWNLOAD_DIR)
        
        # Summary
        total_downloaded = sum(len(r['downloaded']) for r in results)
        total_skipped = sum(len(r['skipped']) for r in results)
        print(f"\n" + "="*50)
        print(f"? ALL DONE!")
        print(f"?? Total: {total_downloaded} photos downloaded, {total_skipped} skipped")
        print(f"?? Photos organized in '{DOWNLOAD_DIR}' folder by person name.")
        for r in results:
            print(f"   - {r['folder']}: {len(r['downloaded'])} matches")
    else:
        # Original behavior - scan entire drive
        print("\n??  No SHARED_FOLDER_ID set. Scanning entire Drive (slower)...")
        print("?? Tip: Set SHARED_FOLDER_ID to scan a specific folder.")
        photo_paths = download_matching_photos(service, DOWNLOAD_DIR, reference_encodings)
        print(f"\n? Done! Downloaded {len(photo_paths)} matching photo(s).")
        print(f"Photos are organized in '{DOWNLOAD_DIR}' folder by person name.")


if __name__ == '__main__':
    main()
