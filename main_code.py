from roboflow import Roboflow

rf = Roboflow(api_key="ZhZaMoPKIieaDQ4V7C9I")
project = rf.workspace("elice-hbahd").project("hole_inspection")
dataset = project.version(13).download("yolov8")
# %pip install ultralytics
import ultralytics
ultralytics.checks()

from glob import glob

train_img_list = glob('/Users/hyucksamkwon/project/streamlit_web_deploy_test/Hole_inspection-13/train/images/*.jpg')
test_img_list = glob('/Users/hyucksamkwon/project/streamlit_web_deploy_test/Hole_inspection-13/test/images/*.jpg')
vaild_img_list = glob('/Users/hyucksamkwon/project/streamlit_web_deploy_test/Hole_inspection-13/valid/images/*.jpg')

print(len(train_img_list), len(test_img_list), len(vaild_img_list))

import yaml

with open('/Users/hyucksamkwon/project/streamlit_web_deploy_test/Hole_inspection-13/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('/Users/hyucksamkwon/project/streamlit_web_deploy_test/Hole_inspection-13/test.txt', 'w') as f:
    f.write('\n'.join(test_img_list) + '\n')

with open('/Users/hyucksamkwon/project/streamlit_web_deploy_test/Hole_inspection-13/vaild.txt', 'w') as f:
    f.write('\n'.join(vaild_img_list) + '\n')

from IPython.core.magic import register_line_cell_magic

# @register_line_cell_magic
# def writetemplate(line, cell):
#     with open(line, 'w') as f:
#         f.write(cell.format(**globals()))

with open('/Users/hyucksamkwon/project/streamlit_web_deploy_test/Hole_inspection-13/data.yaml', 'r') as stream:
    num_calsses = str(yaml.safe_load(stream)['nc'])

from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
model.train(data='/Users/hyucksamkwon/project/streamlit_web_deploy_test/Hole_inspection-13/data.yaml', epochs=100) 

import locale
locale.getpreferredencoding = lambda: "UTF-8"

from IPython.display import Image, clear_output