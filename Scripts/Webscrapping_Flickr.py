import flickrapi
import urllib
import os

# Klucz API i tajny klucz
api_key = "86cc902daf86a8146c5e61d49f86301d"
api_secret = "fae494aed75403ec"

# Inicjalizacja obiektu FlickrAPI
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

# Wykonanie zapytania, aby uzyskać identyfikator użytkownika na podstawie nazwy użytkownika
user = flickr.people.findByUsername(username='agnieszka186')
user_id = user['user']['nsid']

# Pobieranie identyfikatorów galerii
photosets = flickr.photosets.getList(user_id=user_id)
gallery_ids = {}
for photoset in photosets['photosets']['photoset']:
    if photoset['title']['_content'] == 'Landscapes':
        gallery_ids['Landscapes'] = photoset['id']
    elif photoset['title']['_content'] == 'Night_photos':
        gallery_ids['Night_photos'] = photoset['id']

# Pobieranie i zapisywanie zdjęć
for gallery_name, gallery_id in gallery_ids.items():
    photos = flickr.photosets.getPhotos(photoset_id=gallery_id, user_id=user_id, per_page=50)

    # Tworzenie folderu dla zdjęć danej galerii
    folder_path = f"downloaded_photos/{gallery_name}"
    os.makedirs(folder_path, exist_ok=True)

    # Pobieranie i zapisywanie zdjęć
    for i, photo in enumerate(photos['photoset']['photo']):
        # Pobieranie adresu URL oryginalnego rozmiaru zdjęcia
        sizes = flickr.photos.getSizes(photo_id=photo['id'])
        original_size = next((size for size in sizes['sizes']['size'] if size['label'] == 'Original'), None)
        image_url = original_size['source']

        # Pobieranie nazwy pliku z adresu URL
        filename = f"{photo['id']}.jpg"
        file_path = os.path.join(folder_path, filename)

        # Pobieranie i zapisywanie obrazu na dysku
        urllib.request.urlretrieve(image_url, file_path)

        print(f"Downloaded: {filename}")

    print(f"Downloading photos from gallery completed '{gallery_name}'.")

print("Photos download completed")
