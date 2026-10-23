from flask import Blueprint, request, jsonify, render_template, current_app
from datetime import datetime
from app import db
from app.models.fraud_detector import FraudDetector
import traceback

main = Blueprint('main', __name__)


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE SQLAlchemy — Table de journalisation
# ══════════════════════════════════════════════════════════════════════════════

class PredictionLog(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    timestamp        = db.Column(db.DateTime, default=datetime.utcnow)
    amount           = db.Column(db.Float)
    probability      = db.Column(db.Float)
    is_fraud         = db.Column(db.Boolean)
    risk_level       = db.Column(db.String(20))
    transaction_data = db.Column(db.Text)

    def to_dict(self):
        return {
            'id'          : self.id,
            'timestamp'   : self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'amount'      : self.amount,
            'probability' : self.probability,
            'is_fraud'    : self.is_fraud,
            'risk_level'  : self.risk_level
        }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER : Récupérer le détecteur avec les deux scalers
# ══════════════════════════════════════════════════════════════════════════════

def get_detector():
    """Construit le FraudDetector à partir de la config Flask (modèle + 2 scalers)."""
    model          = current_app.config.get('MODEL')
    scaler_amount  = current_app.config.get('SCALER_AMOUNT')
    scaler_time    = current_app.config.get('SCALER_TIME')
    feature_names  = current_app.config.get('FEATURE_NAMES', [])

    if model is None or scaler_amount is None or scaler_time is None:
        return None
    return FraudDetector(model, scaler_amount, scaler_time, feature_names)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER : Conversion sécurisée des données du formulaire
# ══════════════════════════════════════════════════════════════════════════════

def parse_form_data(form_data):
    """
    Convertit TOUTES les valeurs du formulaire HTML en float.
    Le formulaire envoie des chaînes (strings) ; cette fonction les convertit.
    Les clés attendues sont : 'Amount', 'Time', 'V1' ... 'V28'.
    """
    data = {}
    for key, value in form_data.items():
        # Si la valeur est une liste, on prend le premier élément
        if isinstance(value, list):
            value = value[0] if value else ''
        # Ignorer les champs vides
        if value == '' or value is None:
            continue
        try:
            data[key] = float(value)
        except (ValueError, TypeError):
            data[key] = 0.0   # valeur par défaut
    return data


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── GET / — Formulaire de saisie ──────────────────────────────────────────────
@main.route('/')
def index():
    feature_names = current_app.config.get('FEATURE_NAMES', [])
    return render_template('index.html', features=feature_names)


# ── POST /predict — Prédiction principale ─────────────────────────────────────
@main.route('/predict', methods=['POST'])
def predict():
    detector = get_detector()
    if detector is None:
        return jsonify({'error': 'Modèle non disponible. Lancez ml/train.py'}), 503

    try:
        # Récupérer et convertir les données JSON ou formulaire
        if request.is_json:
            raw = request.get_json()
            data = {}
            for k, v in raw.items():
                try:
                    data[k] = float(v)
                except (ValueError, TypeError):
                    data[k] = 0.0
        else:
            data = parse_form_data(request.form)

        if not data:
            return jsonify({'error': 'Aucune donnée fournie'}), 400

        # Appeler le détecteur (il normalisera Amount et Time grâce aux scalers)
        result = detector.predict(data)

        # Journaliser dans SQLite
        log = PredictionLog(
            amount           = data.get('Amount', 0.0),
            probability      = result['probability'],
            is_fraud         = result['is_fraud'],
            risk_level       = result['risk_level'],
            transaction_data = str(data)
        )
        db.session.add(log)
        db.session.commit()

        # Réponse HTML ou JSON
        if not request.is_json:
            return render_template('result.html', result=result, data=data)
        return jsonify(result), 200

    except Exception as e:
        print("=" * 50)
        print("ERREUR dans /predict :")
        print(traceback.format_exc())
        print("=" * 50)
        return jsonify({'error': str(e)}), 500


# ── GET /stats — Statistiques du modèle ───────────────────────────────────────
@main.route('/stats')
def stats():
    metrics    = current_app.config.get('MODEL_METRICS', {})
    model_name = current_app.config.get('MODEL_NAME', 'N/A')

    total  = PredictionLog.query.count()
    frauds = PredictionLog.query.filter_by(is_fraud=True).count()

    return jsonify({
        'model_name'       : model_name,
        'metrics'          : metrics,
        'total_predictions': total,
        'fraud_detected'   : frauds,
        'normal_detected'  : total - frauds,
        'fraud_rate'       : round(frauds / total * 100, 2) if total > 0 else 0
    })


# ── GET /history — 50 dernières prédictions ───────────────────────────────────
@main.route('/history')
def history():
    logs = PredictionLog.query \
        .order_by(PredictionLog.timestamp.desc()) \
        .limit(50).all()
    return jsonify([log.to_dict() for log in logs])


# ── GET /dashboard — Dashboard admin ──────────────────────────────────────────
@main.route('/dashboard')
def dashboard():
    total  = PredictionLog.query.count()
    frauds = PredictionLog.query.filter_by(is_fraud=True).count()
    recent = PredictionLog.query \
        .order_by(PredictionLog.timestamp.desc()) \
        .limit(10).all()

    return render_template(
        'dashboard.html',
        total      = total,
        frauds     = frauds,
        normal     = total - frauds,
        recent     = [r.to_dict() for r in recent],
        model_name = current_app.config.get('MODEL_NAME', 'N/A'),
        metrics    = current_app.config.get('MODEL_METRICS', {}),
        iso_available = current_app.config.get('ISO_FOREST') is not None
    )