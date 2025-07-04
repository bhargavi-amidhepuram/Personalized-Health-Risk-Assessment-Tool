import numpy as np

# Population averages for missing data imputation
POP_AVG = {
    'age': 40,
    'bmi': 25,
    'systolic_bp': 120,
    'smoker': 0,
    'family_history': 0,
    'exercise_freq': 3,
}

MODELS = {
    'Diabetes': {
        'intercept': -5.0,
        'coeffs': {
            'age': 0.04,
            'bmi': 0.1,
            'systolic_bp': 0.02,
            'smoker': 0.3,
            'family_history': 0.6,
            'exercise_freq': -0.15,
        },
        'threshold': 0.2,
        'prevention': [
            "Control blood sugar through diet and regular exercise.",
            "Regular fasting glucose or HbA1c testing.",
            "Limit sugary/processed foods.",
            "Increase physical activity to 150+ minutes/week.",
        ],
        'care': [
            "Adhere to prescribed diabetes medication.",
            "Monitor blood sugar regularly.",
            "Consult endocrinologist for management.",
            "Attend diabetes education programs.",
        ],
    },
    'Heart Disease': {
        'intercept': -6.0,
        'coeffs': {
            'age': 0.05,
            'bmi': 0.07,
            'systolic_bp': 0.04,
            'smoker': 0.7,
            'family_history': 0.8,
            'exercise_freq': -0.1,
        },
        'threshold': 0.25,
        'prevention': [
            "Maintain healthy blood pressure and cholesterol.",
            "Avoid tobacco and excess alcohol.",
            "Adopt heart-healthy diet.",
            "Regular aerobic exercise.",
        ],
        'care': [
            "Follow cardiologist-prescribed medications.",
            "Attend regular heart health checkups.",
            "Manage stress with relaxation techniques.",
            "Seek urgent care for chest pain.",
        ],
    },
    'Obesity': {
        'intercept': -4.5,
        'coeffs': {
            'age': 0.02,
            'bmi': 0.15,
            'systolic_bp': 0.015,
            'smoker': 0.1,
            'family_history': 0.3,
            'exercise_freq': -0.2,
        },
        'threshold': 0.3,
        'prevention': [
            "Balanced, calorie-controlled diet.",
            "Increase daily physical activity.",
            "Reduce sedentary time.",
            "Consult nutritionist as needed.",
        ],
        'care': [
            "Consult healthcare provider for weight management.",
            "Consider behavioral or medical interventions.",
            "Monitor for diabetes and heart disease.",
            "Maintain consistent lifestyle changes.",
        ],
    }
}

def logistic(x):
    return 1 / (1 + np.exp(-x))

def impute_missing(value, feature_name):
    if value is None or value == '':
        return POP_AVG[feature_name], True
    return value, False

def calculate_risk(inputs, model):
    x = model['intercept']
    for feat, coef in model['coeffs'].items():
        x += coef * inputs[feat]
    return logistic(x)
