Some Notes:
1. The Safetensors model was to large for the github project and as a result had to be uploaded to my onedrive. The link to the file is available below, with instructions.
2. I have also included the colab notebooks I used for training the models in this repository.
   These include:
     BILSTM.ipynb
     LSVM.ipynb
     ROBERTA.ipynb
     DISTILBERT.ipynb
      
Setup and Installation

1. Clone the repository
git clone https://github.com/plrushe/Honours-Code.git
cd project
2. Create a virtual environment (Windows)
py -3.11 -m venv venv
venv\Scripts\activate
3. Verify Python version
python --version

Expected:

Python 3.10 or higher
4. Install dependencies
pip install -r requirements.txt
install model.safetensors from cloud (File too large for github)
https://caledonianac-my.sharepoint.com/:u:/g/personal/prushe300_caledonian_ac_uk/IQBZlGC4gsMmTK9z-ybO_ULAAU9QnTlrEsslxQayCruNVtI?e=J2Octb
and place it in the distilbert_depression_classifier folder.

Run the application from the project root:

python -m src.app
