from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import joblib
import os
import sys

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'votre-cle-secrete-changez-moi'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    # --- Chargement du modèle ---
    model_path = os.path.join(app.root_path, 'models', 'saved_model.pkl')
    print(f"[DEBUG] app.root_path = {app.root_path}")
    print(f"[DEBUG] model_path = {model_path}")
    print(f"[DEBUG] exists = {os.path.exists(model_path)}")

    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            print(f"[DEBUG] Clés trouvées : {list(model_data.keys())}")

            app.config['MODEL'] = model_data['model']
            app.config['MODEL_NAME'] = model_data['model_name']
            app.config['MODEL_METRICS'] = model_data.get('metrics', {})
            app.config['FEATURE_NAMES'] = model_data['feature_names']
            app.config['SCALER_AMOUNT'] = model_data['scaler_amount']
            app.config['SCALER_TIME'] = model_data['scaler_time']

            print(f"✅ Modèle chargé : {model_data['model_name']}")
            if app.config['SCALER_AMOUNT'] is None or app.config['SCALER_TIME'] is None:
                print("⚠️ Scalers manquants dans saved_model.pkl")
            else:
                print("✅ Scalers (Amount / Time) chargés avec succès.")
        except Exception as e:
            print(f"❌ Erreur au chargement : {e}")
            import traceback
            traceback.print_exc()
            app.config['MODEL'] = None
    else:
        print(f"❌ Fichier introuvable : {model_path}")
        app.config['MODEL'] = None

    from app.routes import main
    app.register_blueprint(main)

    with app.app_context():
        db.create_all()

    return app