import numpy as np
import joblib
from flask import Flask, request, render_template_string

app = Flask(__name__)

scaler = joblib.load('plots/scaler.pkl')
feat_idx = joblib.load('plots/feat_idx.pkl')
common_features = joblib.load('plots/common_features.pkl')
models = {}
for name in ['RF', 'XGB', 'LR', 'LGBM']:
    models[name] = joblib.load(f'plots/model_{name}.pkl')

FIELDS = [
    ('BIPS', 'BIPS (score)', 'number', '14', '152', '0.1'),
    ('Eating_in_front_of_screen', 'Eating in front of screen', 'select', None, None, None),
    ('MPA_time', 'MPA time (h)', 'number', '0', '5', '0.01'),
    ('Monthly_income', 'Monthly income (RMB)', 'select_income', None, None, None),
    ('PMT', 'PMT (score)', 'number', '73', '152', '0.1'),
    ('Parity', 'Parity (times)', 'number', '0', '10', '1'),
    ('Pre_pregnancy_BMI', 'Pre-pregnancy BMI (kg/m²)', 'select_bmi', None, None, None),
    ('Sedentary_time', 'Sedentary time (h)', 'number', '0', '24', '0.1'),
    ('TDEE', 'TDEE (MET-h/daily)', 'number', '10', '60', '0.01'),
    ('Weekly_frequency_of_SSB_Desserts_WFF', 'Weekly frequency of SSB/Desserts/WFF', 'select_freq', None, None, None),
]

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Excessive Gestational Weight Gain Predictor</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f5f5f5;color:#333;font-size:14px}
.wrap{max-width:580px;margin:40px auto;padding:0 20px}
h1{font-size:18px;font-weight:600;margin-bottom:20px;color:#222}
.row{display:flex;align-items:center;margin-bottom:12px}
.row label{width:260px;font-size:13px;color:#444;flex-shrink:0}
.row input,.row select{flex:1;height:32px;border:1px solid #ccc;padding:0 8px;font-size:13px;background:#fff;outline:none}
.row input:focus,.row select:focus{border-color:#666}
.row select{cursor:pointer}
.mrow{margin-top:16px;margin-bottom:16px;display:flex;align-items:center}
.mrow label{width:260px;font-size:13px;color:#444;flex-shrink:0}
.mrow select{flex:1;height:32px;border:1px solid #ccc;padding:0 8px;font-size:13px;background:#fff;outline:none;cursor:pointer}
button{display:block;width:100%;height:36px;background:#333;color:#fff;border:none;font-size:14px;cursor:pointer;margin-top:8px}
button:hover{background:#555}
.result{margin-top:20px;padding:16px;background:#fff;border:1px solid #ddd}
.result h2{font-size:15px;font-weight:600;margin-bottom:10px}
.result p{font-size:13px;line-height:1.8;color:#333}
.risk-high{color:#c0392b;font-weight:600}
.risk-low{color:#27ae60;font-weight:600}
</style>
</head>
<body>
<div class="wrap">
<h1>Excessive Gestational Weight Gain Predictor</h1>
<form method="POST">
{% for key, label, ftype, mn, mx, st in fields %}
<div class="row">
<label>{{ label }}</label>
{% if ftype == 'select' %}
<select name="{{ key }}">
<option value="0" {% if vals.get(key)=='0' %}selected{% endif %}>No (0)</option>
<option value="1" {% if vals.get(key)=='1' %}selected{% endif %}>Yes (1)</option>
</select>
{% elif ftype == 'select_income' %}
<select name="{{ key }}">
<option value="1" {% if vals.get(key)=='1' %}selected{% endif %}>≤4000 (1)</option>
<option value="2" {% if vals.get(key)=='2' %}selected{% endif %}>4001-8000 (2)</option>
<option value="3" {% if vals.get(key)=='3' %}selected{% endif %}>8001-15000 (3)</option>
<option value="4" {% if vals.get(key)=='4' %}selected{% endif %}>>15000 (4)</option>
</select>
{% elif ftype == 'select_bmi' %}
<select name="{{ key }}">
<option value="1" {% if vals.get(key)=='1' %}selected{% endif %}><18.5 (1)</option>
<option value="2" {% if vals.get(key)=='2' %}selected{% endif %}>18.5-24.0 (2)</option>
<option value="3" {% if vals.get(key)=='3' %}selected{% endif %}>24.0-28.0 (3)</option>
<option value="4" {% if vals.get(key)=='4' %}selected{% endif %}>≥28.0 (4)</option>
</select>
{% elif ftype == 'select_freq' %}
<select name="{{ key }}">
<option value="1" {% if vals.get(key)=='1' %}selected{% endif %}><1 (1)</option>
<option value="2" {% if vals.get(key)=='2' %}selected{% endif %}>1-2 (2)</option>
<option value="3" {% if vals.get(key)=='3' %}selected{% endif %}>3-4 (3)</option>
<option value="4" {% if vals.get(key)=='4' %}selected{% endif %}>≥5 (4)</option>
</select>
{% else %}
<input type="number" name="{{ key }}" min="{{ mn }}" max="{{ mx }}" step="{{ st }}" value="{{ vals.get(key, '') }}" required>
{% endif %}
</div>
{% endfor %}
<div class="mrow">
<label>Model</label>
<select name="model_name">
{% for m in model_names %}
<option value="{{ m }}" {% if vals.get('model_name')==m %}selected{% endif %}>{{ m }}{% if m=='XGB' %} (Best){% endif %}</option>
{% endfor %}
</select>
</div>
<button type="submit">Predict</button>
</form>
{% if result is not none %}
<div class="result">
<h2>Prediction Result ({{ result.model }})</h2>
<p>Probability of excessive weight gain: <strong>{{ result.prob }}</strong></p>
<p>Classification: <span class="{{ 'risk-high' if result.label == 1 else 'risk-low' }}">{{ 'High Risk (Excessive Weight Gain)' if result.label == 1 else 'Low Risk (Normal Weight Gain)' }}</span></p>
</div>
{% endif %}
</div>
</body>
</html>'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    vals = {}
    if request.method == 'POST':
        vals = dict(request.form)
        model_name = vals.get('model_name', 'XGB')
        full_input = np.zeros((1, 25))
        for feat in common_features:
            col_idx = feat_idx[common_features.index(feat)]
            full_input[0, col_idx] = float(vals[feat])
        scaled = scaler.transform(full_input)
        X_input = scaled[:, feat_idx]
        model = models[model_name]
        prob = model.predict_proba(X_input)[0, 1]
        label = int(prob >= 0.5)
        result = {'model': model_name, 'prob': f'{prob:.4f}', 'label': label}
    return render_template_string(HTML, fields=FIELDS, vals=vals,
                                  result=result, model_names=['XGB', 'RF', 'LGBM', 'LR'])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
