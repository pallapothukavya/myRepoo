import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI(title="White Blood Cell & Animal Classification Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path (relative to the project root)
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db.sqlite3")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "multiclass.h5")

# Class names from Django app
CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Load model once at startup
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class UserRegister(BaseModel):
    name: str
    loginid: str
    password: str
    mobile: str
    email: EmailStr
    locality: str = ""
    address: str = ""
    city: str = ""
    state: str = ""

class UserLogin(BaseModel):
    loginid: str
    password: str

@app.get("/")
async def root():
    return {"message": "FastAPI Backend is running"}

@app.post("/register")
async def register(user: UserRegister):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT id FROM user_registrations WHERE loginid = ? OR email = ? OR mobile = ?", 
                       (user.loginid, user.email, user.mobile))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="User with this loginid, email or mobile already exists")
        
        # Insert user
        cursor.execute("""
            INSERT INTO user_registrations (name, loginid, password, mobile, email, locality, address, city, state, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user.name, user.loginid, user.password, user.mobile, user.email, user.locality, user.address, user.city, user.state, 'waiting'))
        
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Registration successful! Waiting for admin activation."}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/login")
async def login(user: UserLogin):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, status FROM user_registrations WHERE loginid = ? AND password = ?", 
                       (user.loginid, user.password))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=401, detail="Invalid login ID or password")
        
        if row[3] != "activated":
            raise HTTPException(status_code=401, detail="Account not activated by admin")
            
        return {
            "status": "success",
            "message": "Login successful",
            "user": {
                "id": row[0],
                "name": row[1],
                "email": row[2]
            }
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model not loaded")
    
    try:
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image (same as in Django views.py)
        img = img.resize((160, 120))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        confidence = float(np.max(prediction))
        predicted_index = int(np.argmax(prediction))
        
        threshold = 0.98
        if confidence < threshold:
            result = "This is not a valid White Blood Cell/Target image"
            status = "invalid"
        else:
            result = CLASS_NAMES[predicted_index]
            status = "success"
            
        return {
            "status": status,
            "predicted_class": result,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
