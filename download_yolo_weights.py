import urllib.request
url = "https://huggingface.co/chanieew/yolov8n-fashion-clothes/resolve/main/best.pt?download=true"
urllib.request.urlretrieve(url, "yolov8n-fashion.pt")
