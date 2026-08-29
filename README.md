# 📱 Smartphone Price Predictor

Smartphone specs वरून price predict करणारा ML app — Gradient Boosting Regressor + Streamlit.

## Folder Structure

```
smartphone-price-app/
├── app.py                # Streamlit app
├── train.py               # Training script
├── requirements.txt
├── .gitignore
├── README.md
└── Smartphones.csv        # (तुझी data file — इथे टाक)
```

## 1️⃣ Local वर आधी test कर

```bash
cd smartphone-price-app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Smartphones.csv या फोल्डरमध्ये टाक, मग:
python train.py               # model_pipeline.joblib + metadata.json तयार होतील

streamlit run app.py          # local browser मध्ये app उघडेल
```

`train.py` run केल्यावर दोन नवीन फाइल्स तयार होतात:
- `model_pipeline.joblib` — trained pipeline (preprocessing + model)
- `metadata.json` — form बनवण्यासाठी categories/ranges

या दोन्ही फाइल्स **GitHub वर push करायच्या आहेत** (मॉडेल फाइल), कारण Streamlit Cloud वर `train.py` run होत नाही आपोआप — फक्त `app.py` चालतो.

## 2️⃣ GitHub वर push कर

```bash
git init
git add .
git commit -m "Smartphone price predictor - initial commit"

# GitHub वर आधी नवीन empty repo बनव (github.com/new), मग:
git branch -M main
git remote add origin https://github.com/<तुझं-username>/smartphone-price-app.git
git push -u origin main
```

⚠️ जर `Smartphones.csv` मोठी फाइल असेल (>100MB), ती GitHub वर push होणार नाही — अशावेळी फक्त `model_pipeline.joblib` आणि `metadata.json` push कर, CSV `.gitignore` मध्ये टाक. `model_pipeline.joblib` सुद्धा साधारण काही MB च्या आत असतो, त्यामुळे बहुतांश वेळा प्रॉब्लेम येत नाही.

## 3️⃣ Streamlit Community Cloud वर deploy कर

1. https://share.streamlit.io वर जा आणि GitHub अकाउंटने login कर
2. **"New app"** क्लिक कर
3. Repository: तुझं `smartphone-price-app` repo निवड
4. Branch: `main`, Main file path: `app.py`
5. **Deploy** क्लिक कर — 1-2 मिनिटांत लाईव्ह URL मिळेल (उदा. `yourapp.streamlit.app`)

## Notes

- App ला `model_pipeline.joblib` आणि `metadata.json` repo मध्ये असणं गरजेचं आहे — नाहीतर app error देईल ("Run train.py first").
- मॉडेल बदलायचा/retrain करायचा असेल तर local वर `train.py` परत run कर, नवीन `.joblib`/`.json` फाइल्स commit-push कर, Streamlit Cloud आपोआप redeploy होईल.
