import os
import urllib.request
from urllib.error import HTTPError, URLError
import zipfile
import shutil

url = "https://cdn.intra.42.fr/document/document/42144/leaves.zip"
zip = "data.zip"
dataFolder = "data"

def downloadZip():
	if os.path.exists(zip):
		os.remove(zip)
	try:
		print("Downloading Dataset Zip")
		urllib.request.urlretrieve(url, zip)
		print("Download Completed")
		unzipData()
	except HTTPError as e:
		print("Error: HTTP Error")
	except URLError as e:
		print("Error: URL Error")
	except Exception as e:
		print("Error: Unowkn Error")

def unzipData():
	try:
		with zipfile.ZipFile(zip, "r") as z:
			if z.testzip() is not None:
				print("Zip file is Corrupted")
			else:
				z.extractall()
				print("Unzip Successful")
				replaceData()
	except zipfile.BadZipFile:
		print("Error: The file is not a zip file or is corrupted.")
	except FileNotFoundError:
		print("Error: The zip file was not found.")
	except PermissionError:
		print("Error: You don't have permission to write to that folder.")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")

def replaceData():
	if os.path.exists(dataFolder):
		shutil.rmtree(dataFolder)
		print("Old data removed")
	os.rename("images", dataFolder)
	os.remove(zip)

downloadZip()










