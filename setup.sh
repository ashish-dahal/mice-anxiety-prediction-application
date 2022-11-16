python -m venv $(pwd)/venv
source $(pwd)/venv/Scripts/activate
pip install moviepy==1.0.3
pip install imageio==2.9.0
pip install python-rapidjson==1.4
pip install pywinpty==1.1.1
pip install streamlit
pip install xlsxwriter
pip install qtconsole==5.2.2
pip install sleap==1.1.5
pip uninstall -y opencv-python-headless
pip install git+https://github.com/opencv/opencv-python