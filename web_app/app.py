from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import re
import os
from pathlib import Path

app = Flask(__name__, static_folder='static')
model = tf.keras.models.load_model('../food_recognition_model.h5')

with open("../classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

API_KEY = 'AIzaSyDsf3Zg0mPPSyb7F5YmXq6YQwCOUV2Mh94'
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        img = Image.open(image_file)
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        
        confidence = float(np.max(prediction)) * 100
        confidence_percent = round(confidence, 2)
        
        return jsonify({
            'food': predicted_class,
            'confidence': confidence_percent
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/get-food-info', methods=['POST'])
def get_food_info():
    try:
        data = request.get_json()
        food_name = data.get('foodName')

        if not food_name:
            return jsonify({'error': 'No food name provided'}), 400

        prompt = f"""
            Analyze {food_name} and provide this EXACT structure:
            1. **Title**: [Dish Name]
            2. **Ingredients** (with 1-5 health ratings):
                - [Ingredient] (X/5): [Description]
            3. **Nutrition** per serving:
               Protein: [XX]g
               Carbohydrates: [XX]g
               Fats: [XX]g
               Sugar: [XX]g
               Sodium: [XX]mg
               Fiber: [XX]g
               Vitamin C: [XX]% DV
               Calcium: [XX]% DV
               Iron: [XX]% DV
            4. **Health Assessment**:
               Healthiness: [1-sentence assessment]
               Suggestions: [2 practical improvements]
            Use strict markdown formatting. Include all sections.
        """

        body = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }

        response = requests.post(GEMINI_URL, json=body, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = response.json()

        if not result.get('candidates'):
            return jsonify({'error': 'No response from Gemini'}), 500

        text = result['candidates'][0]['content']['parts'][0]['text']
        
        # Parse nutrition
        nutrition = {}
        try:
            nutrition_block = text.split('**Nutrition**')[-1].split('**Health Assessment**')[0]
            patterns = {
                'protein': r'Protein:\s*([\d.]+)\s*g',
                'carbs': r'Carbohydrates?:\s*([\d.]+)\s*g',
                'fats': r'Fats:\s*([\d.]+)\s*g',
                'sugar': r'Sugar:\s*([\d.]+)\s*g',
                'sodium': r'Sodium:\s*([\d.]+)\s*mg',
                'fiber': r'Fiber:\s*([\d.]+)\s*g',
                'vitamin_c': r'Vitamin C:\s*([\d.]+)%',
                'calcium': r'Calcium:\s*([\d.]+)%',
                'iron': r'Iron:\s*([\d.]+)%'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, nutrition_block, re.IGNORECASE)
                nutrition[key] = match.group(1) if match else 'N/A'
                
        except Exception as e:
            print("Nutrition parsing error:", e)
            nutrition = {key: 'N/A' for key in patterns.keys()}

        # Parse ingredients
        ingredients = []
        try:
            ingredients_block = text.split('**Ingredients**')[1].split('**Recipe**')[0]
            for line in ingredients_block.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    clean_line = re.sub(r'^-\s*', '', line)
                    rating_match = re.search(r'\((\d)/5\)', clean_line)
                    if rating_match:
                        name = clean_line.split('(')[0].strip()
                        rating = int(rating_match.group(1))
                    else:
                        name = clean_line
                        rating = 3
                    ingredients.append({'name': name, 'rating': rating})
        except Exception as e:
            print("Ingredient parsing error:", e)

        # Parse health assessment
        healthiness = ""
        suggestion = ""
        try:
            health_block = text.split('**Health Assessment**')[1].strip()
            healthiness = health_block.split('Suggestions:')[0].replace('Healthiness:', '').strip()
            suggestion = health_block.split('Suggestions:')[1].strip() if 'Suggestions:' in health_block else ""
        except Exception as e:
            print("Health assessment error:", e)

        return jsonify({
            'food': food_name,
            'nutrition': nutrition,
            'ingredients': ingredients,
            'healthiness': healthiness,
            'suggestion': suggestion
        })

    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Gemini API error'}), 500
    except Exception as e:
        return jsonify({'error': 'Processing error'}), 500



COMMUNITY_DATA_DIR = Path("community_dataset")
DISH_LIST_FILE = COMMUNITY_DATA_DIR / "dish_list.txt"

@app.route('/save-image', methods=['POST'])
def save_image():
    try:
        if 'image' not in request.files or not request.files['image']:
            return jsonify({'success': False, 'error': 'No image provided'})
            
        dish_name = request.form.get('dish', '').strip().lower()
        if not dish_name:
            return jsonify({'success': False, 'error': 'No dish name provided'})
        
        # Sanitize dish name
        dish_name = re.sub(r'[^a-zA-Z0-9_]', '_', dish_name)
        if not dish_name:
            return jsonify({'success': False, 'error': 'Invalid dish name'})

        # Create community dataset dir if needed
        COMMUNITY_DATA_DIR.mkdir(exist_ok=True)
        
        # Create dish directory
        dish_dir = COMMUNITY_DATA_DIR / dish_name
        dish_dir.mkdir(exist_ok=True)
        
        # Save image
        image = request.files['image']
        image_path = dish_dir / f"{len(list(dish_dir.glob('*')))+1}.jpg"
        image.save(image_path)
        
        # Update dish list
        with open(DISH_LIST_FILE, 'a+') as f:
            f.seek(0)
            existing_dishes = {line.strip().lower() for line in f}
            if dish_name not in existing_dishes:
                f.write(dish_name + '\n')
        
        return jsonify({'success': True})
        
    except Exception as e:
        print("Save image error:", e)
        return jsonify({'success': False, 'error': 'Server error saving image'})

if __name__ == '__main__':
     app.run(debug=True, host='127.0.0.1', port=8000)