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

if __name__ == "__main__":
    app.run(debug=True)
