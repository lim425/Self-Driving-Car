from PIL import ImageFont
import visualkeras
import os
from tensorflow.keras.models import load_model

def Vis_CNN(model):
    font = ImageFont.load_default()
    model.add(visualkeras.SpacingDummyLayer(spacing=100))
    visualkeras.layered_view(model, to_file='/home/junyi/potbot_venv_ws/src/sdc_venv_pkg/sdc_venv_pkg/data/Visual_CNN.png',legend=True, font=font,scale_z=2).show()  # font is optional!


def main():

    model = load_model(os.path.abspath('/home/junyi/potbot_venv_ws/src/sdc_venv_pkg/sdc_venv_pkg/data/saved_model_Ros2_5_Sign.h5'),compile=False)
    Vis_CNN(model)
    

if __name__ == '__main__':
	main()