from google.colab import files
uploaded = files.upload()


!pip install streamlit pyngrok
from pyngrok import ngrok
!ngrok authtoken 2ivREWYKjLFSnTDuRJHLcQnkdl9_2mxGJr9ogU7TYMuAhHRwK
ngrok_tunnel = ngrok.connect(8501)
print('Public URL:', ngrok_tunnel.public_url)
!streamlit run app.py &>/dev/null&
