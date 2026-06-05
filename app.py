import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        .carousel-row { display: flex; overflow-x: auto; gap: 10px; padding: 10px 0; }
        .carousel-item { height: 60px; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 8 * 8, 10) 

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

@st.cache_resource
def load_models():
    cnn = SimpleCNN()
    cnn.load_state_dict(torch.load('models/cnn_digits.pt', map_location='cpu'))
    cnn.eval()
    return YOLO('models/best.pt'), cnn

yolo_model, cnn_model = load_models()

file = st.file_uploader("Завантажте фото", type=["jpg", "jpeg", "png"])

if file:
    img = np.array(Image.open(file))
    st.image(img, width=400)
    
    obb = yolo_model(img)[0].obb
    
    if obb is not None and len(obb.xyxyxyxy) > 0:
        pts = obb.xyxyxyxy[0].cpu().numpy()
        
        def get_x(point):
            return point[0] 

        def get_y(point):
            return point[1] 
        
        pts = sorted(pts, key=get_y)

        top_points = sorted(pts[:2], key=get_x)
        top_left = top_points[0]
        top_right = top_points[1]
        
        bottom_points = sorted(pts[2:], key=get_x)
        bottom_left = bottom_points[0]
        bottom_right = bottom_points[1]
        
        src = np.float32([top_left, top_right, bottom_right, bottom_left])
        
        for i in range(8):
            crop = warped[:, i*step:(i+1)*step]
            mx, my = int(step * 0.15), int(h_w * 0.1)
            crop = crop[my:h_w-my, mx:step-mx]
            
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            tensor = transforms.ToTensor()(cv2.resize(gray, (32, 32))).unsqueeze(0)
            
            with torch.no_grad():
                result_text += str(torch.argmax(cnn_model(tensor)).item())
            
            buf = BytesIO()
            Image.fromarray(crop).save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode()
            carousel_html += f"<img src='data:image/png;base64,{img_str}' class='carousel-item'>"
            
        st.markdown(carousel_html + "</div>", unsafe_allow_html=True)
        st.success(result_text)
    else:
        st.error("Лічильник не знайдено")
