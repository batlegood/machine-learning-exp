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
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    
    data = pd.read_csv("dataset_regresion_logistica (1).csv")

    
    X = data[["edad", "ingreso_mensual", "visitas_web_mes","tiempo_sitio_min", "compras_previas", "descuento_usado"]]
    
    y = data["target"]


    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred).tolist()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)


    num_records=len(data)
    independent_vars = X.columns.tolist()
    target_var = "target"
    class_meanings = {0: "No compra", 1: "Compra"}
    train_size=len(X_train)
    test_size=len(X_test)

    plt.figure(figsize=(6,4))
    plt.scatter(data["edad"], data["ingreso_mensual"], c=y, cmap="bwr", alpha=0.6)
    plt.title("Dataset Visualization: Age vs Monthly Income")
    plt.xlabel("Age")
    plt.ylabel("Monthly Income")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='No Purchase (0)', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Purchase (1)', markerfacecolor='red', markersize=8)
    ])
    
    plt.tight_layout()
    plt.savefig("static/plot.png")
    plt.close()


        
    if request.method == "POST":
        edad = int(request.form["edad"])
        ingreso = int(request.form["ingreso_mensual"])
        visitas = int(request.form["visitas_web_mes"])
        tiempo = float(request.form["tiempo_sitio_min"])
        compras = int(request.form["compras_previas"])
        descuento = int(request.form["descuento_usado"])

        prediction = model.predict([[edad, ingreso, visitas, tiempo, compras, descuento]])[0]
        prediction = "Compra (1)" if prediction == 1 else "No compra (0)"


    return render_template("logistic_app.html", prediction=prediction, accuracy=accuracy, confusion=confusion, num_records=num_records, independent_vars=independent_vars, target_var=target_var, class_meanings=class_meanings, train_size=train_size, test_size=test_size, recall=recall, precision=precision)

if __name__ == "__main__":
    app.run(debug=True)

