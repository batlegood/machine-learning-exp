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



@app.route("/linear_regression_concepts")
def linear_regression_concepts():
    return render_template("linear_regression_concepts.html")

@app.route("/logistic_regression_concepts")
def logistic_regression_concepts():
    return render_template("logistic_regression_concepts.html")


@app.route("/logistic_app", methods=["GET", "POST"])
def logistic_app():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Cargar dataset
    data = pd.read_csv("dataset_regresion_logistica (1).csv")
    X = data[["edad", "ingreso_mensual", "visitas_web_mes","tiempo_sitio_min", "compras_previas", "descuento_usado"]]
    y = data["target"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred).tolist()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Info dataset
    num_records = len(data)
    independent_vars = X.columns.tolist()
    target_var = "target"
    class_meanings = {0: "No compra", 1: "Compra"}
    train_size = len(X_train)
    test_size = len(X_test)

    # Gráfico
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

    # Predicción con formulario
    prediction = None
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
        confusion=confusion,
        num_records=num_records,
        independent_vars=independent_vars,
        target_var=target_var,
        class_meanings=class_meanings,
        train_size=train_size,
        test_size=test_size,
        recall=recall,
        precision=precision
    )




@app.route("/clustering")
def clustering():
    from clustering import implementClustering
    clustering_results = implementClustering()
    return render_template("clustering.html", results=clustering_results["results"], summaryClusters=clustering_results["summaryClusters"], centroids=clustering_results["centroids"])


@app.route("/case1")
def case1():
    return render_template("case1.html")

@app.route("/case2")
def case2():
    return render_template("case2.html")

@app.route("/case3")
def case3():
    return render_template("case3.html")

@app.route("/decision_threshold")
def decision_threshold():
    return render_template("decision_threshold.html")
import matplotlib.pyplot as plt

@app.route("/comparison_algorithms")
def comparison_algorithms():
    # Ejemplo de métricas (puedes calcularlas con tus datasets reales)
    algorithms = ["Logistic Regression", "Random Forest", "Linear Regression"]
    accuracies = [0.85, 0.92, 0.78]  # valores de ejemplo

    plt.figure(figsize=(6,4))
    plt.bar(algorithms, accuracies, color=["steelblue","forestgreen","darkorange"])
    plt.title("Comparación de Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("static/comparison_metrics.png")
    plt.close()

    return render_template("comparison_algorithms.html")

@app.route("/kmeans_concepts")
def kmeans_concepts():
    return render_template("kmeans_concepts.html")

import pandas as pd

@app.route("/kmeans_manual")
def kmeans_manual():
    
    df = pd.read_excel("static/KMeans_Iteracion_1_Estudiantes (1).xlsx", sheet_name="Iteracion 1")

    
    table_html = df.to_html(classes="table table-bordered", index=False)

    return render_template("kmeans_manual.html", table=table_html)


@app.route("/case4")
def case4():
    return render_template("case4.html")

@app.route("/random_forest_concepts")
def random_forest_concepts():
    return render_template("random_forest_concepts.html")

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

@app.route("/kmeans_app")
def kmeans_app():
    
    data = pd.read_csv("static/estudiantes.csv")  
    X = data[["Horas_Estudio", "Calificacion"]]

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    
    silhouette = silhouette_score(X_scaled, clusters)

    
    plt.figure(figsize=(6,4))
    plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap="viridis", alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c="red", marker="X", s=200)
    plt.title("Clusters de Estudiantes (K-Means)")
    plt.savefig("static/kmeans_clusters.png")
    plt.close()

    
    summary = pd.DataFrame({
        "Cluster": range(3),
        "Registros": pd.Series(clusters).value_counts().sort_index().values,
        "Centroides": kmeans.cluster_centers_.tolist()
    })

    
    return render_template("kmeans_app.html",
                           table=data.assign(Cluster=clusters).to_html(classes="table table-bordered", index=False),
                           summary=summary.to_html(classes="table table-bordered", index=False),
                           silhouette=silhouette)


@app.route("/random_forest_app", methods=["GET", "POST"])
def random_forest_app():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    # Cargar dataset
    data = pd.read_csv("dataset_random_forest.csv") 
    X = data[["edad","ingreso_mensual","visitas_web_mes","tiempo_sitio_min","compras_previas","descuento_usado"]]
    y = data["target"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred).tolist()

    prediction = None
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
        "random_forest_app.html",
        prediction=prediction,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        confusion=confusion
    )


if __name__ == "__main__":
    app.run(debug=True)

