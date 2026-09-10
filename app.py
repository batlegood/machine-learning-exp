from flask import Flask, render_template, request
from modelo import predecir_ingresos

app = Flask(__name__)

@app.route("/")
def menu():
    return render_template("menu.html")

@app.route("/machine")
def machine():
    return render_template("machine.html")

@app.route("/ingresos", methods=["GET", "POST"])
def ingresos():
    prediccion = None
    if request.method == "POST":
        clientes = int(request.form["clientes"])
        prediccion = predecir_ingresos(clientes)
    return render_template("ingresos.html", prediccion=prediccion)

# 👇 Aquí ya está fuera de ingresos, alineado en la columna cero
@app.route("/logistic")
def logistic():
    return render_template("logistic.html")

@app.route("/logistic_concepts")
def logistic_concepts():
    return render_template("logistic_concepts.html")

@app.route("/logistic_app", methods=["GET", "POST"])
def logistic_app():
    prediction = None
    accuracy = None
    confusion = None

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, accuracy_score

    # Cargar dataset desde archivo
    data = pd.read_csv("dataset_regresion_logistica (1).csv")

    # Variables independientes
    X = data[["edad", "ingreso_mensual", "visitas_web_mes",
              "tiempo_sitio_min", "compras_previas", "descuento_usado"]]
    # Variable objetivo
    y = data["target"]

    # Entrenar modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Evaluar modelo
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    confusion = confusion_matrix(y, y_pred).tolist()

    # Si el usuario envía datos desde el formulario
    if request.method == "POST":
        edad = int(request.form["edad"])
        ingreso = int(request.form["ingreso_mensual"])
        visitas = int(request.form["visitas_web_mes"])
        tiempo = float(request.form["tiempo_sitio_min"])
        compras = int(request.form["compras_previas"])
        descuento = int(request.form["descuento_usado"])

        prediction = model.predict([[edad, ingreso, visitas, tiempo, compras, descuento]])[0]
        prediction = "Compra (1)" if prediction == 1 else "No compra (0)"

    return render_template(
        "logistic_app.html",
        prediction=prediction,
        accuracy=accuracy,
        confusion=confusion
    )


    
if __name__ == "__main__":
    app.run(debug=True)

