<<<<<<< HEAD
from google.colab import files
uploaded = files.upload()


!pip install streamlit pyngrok
from pyngrok import ngrok
!ngrok authtoken 2ivREWYKjLFSnTDuRJHLcQnkdl9_2mxGJr9ogU7TYMuAhHRwK
ngrok_tunnel = ngrok.connect(8501)
print('Public URL:', ngrok_tunnel.public_url)
!streamlit run app.py &>/dev/null&
=======
from google.colab import files
uploaded = files.upload()


!pip install streamlit pyngrok
from pyngrok import ngrok
!ngrok authtoken 2ivREWYKjLFSnTDuRJHLcQnkdl9_2mxGJr9ogU7TYMuAhHRwK
ngrok_tunnel = ngrok.connect(8501)
print('Public URL:', ngrok_tunnel.public_url)
!streamlit run app.py &>/dev/null&
>>>>>>> c44a98faf6e9c99b83c56c9caf77df8732f2bf28
