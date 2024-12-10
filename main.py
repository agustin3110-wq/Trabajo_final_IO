import numpy as np
import matplotlib.pyplot as plt


def calcular_solucion_optima_con_costos(
    capacidad_contenedor, capacidad_semiremolque, productos, costo_fijo=0
):
    peso_minimo = sum(p["peso"] * p["min_cant"] for p in productos)
    volumen_minimo = sum(p["volumen"] * p["min_cant"] for p in productos)
    beneficio_minimo = sum(
        (p["beneficio"] - p["peso"] - costo_fijo) * p["min_cant"] for p in productos
    )

    if peso_minimo > capacidad_semiremolque or volumen_minimo > capacidad_contenedor:
        print(
            "Las capacidades no son suficientes para cubrir las cantidades mínimas de los productos."
        )
        return 0, (0, 0), {}

    dp = {}
    cantidades_iniciales = {p["nombre"]: p["min_cant"] for p in productos}
    dp[(peso_minimo, volumen_minimo)] = (beneficio_minimo, cantidades_iniciales)

    for producto in productos:
        peso = producto["peso"]
        volumen = producto["volumen"]
        beneficio = producto["beneficio"] - producto["peso"] - costo_fijo
        nombre = producto["nombre"]
        min_cant = producto["min_cant"]
        max_cant = int(
            min(
                (capacidad_semiremolque - peso_minimo) // peso,
                (capacidad_contenedor - volumen_minimo) // volumen,
            )
        )

        nuevos_dp = dp.copy()
        for (peso_acum, volumen_acum), (beneficio_acum, cantidades_acum) in dp.items():
            for k in range(1, max_cant + 1):
                nuevo_peso = peso_acum + peso * k
                nuevo_volumen = volumen_acum + volumen * k
                if (
                    nuevo_peso > capacidad_semiremolque
                    or nuevo_volumen > capacidad_contenedor
                ):
                    break
                nuevo_beneficio = beneficio_acum + beneficio * k
                nueva_cantidades = cantidades_acum.copy()
                nueva_cantidades[nombre] = nueva_cantidades.get(nombre, min_cant) + k
                estado = (nuevo_peso, nuevo_volumen)
                if estado not in nuevos_dp or nuevos_dp[estado][0] < nuevo_beneficio:
                    nuevos_dp[estado] = (nuevo_beneficio, nueva_cantidades)
        dp.update(nuevos_dp)

    max_beneficio, cantidades = max(dp.values(), key=lambda x: x[0])
    estado_optimo = [
        estado for estado, valor in dp.items() if valor[0] == max_beneficio
    ][0]
    peso_utilizado, volumen_utilizado = estado_optimo

    return max_beneficio, (peso_utilizado, volumen_utilizado), cantidades


# Parámetros del problema
capacidad_contenedor = 30  # m³
capacidad_semiremolque = 450  # ton * 10

# Productos disponibles
productos = [
    {
        "peso": 50,
        "volumen": 3,
        "beneficio": 1000,
        "min_cant": 2,
        "nombre": "Producto A",
    },
    {
        "peso": 60,
        "volumen": 5,
        "beneficio": 1200,
        "min_cant": 1,
        "nombre": "Producto B",
    },
    {"peso": 40, "volumen": 2, "beneficio": 800, "min_cant": 3, "nombre": "Producto C"},
    {"peso": 15, "volumen": 1, "beneficio": 400, "min_cant": 5, "nombre": "Producto D"},
]

# Llamada a la función con los parámetros
beneficio_maximo, indices, cantidades = calcular_solucion_optima_con_costos(
    capacidad_contenedor, capacidad_semiremolque, productos
)

# Mostrar resultados
print(f"El beneficio máximo es: ${beneficio_maximo:.2f}")
print(f"Peso utilizado: {indices[0] /10} ton")
print(f"Volumen utilizado: {indices[1]} m³")
print("Cantidades de productos seleccionados:")
for nombre, cantidad in cantidades.items():
    print(f"{nombre}: {cantidad}")


# Analisis de sensibilidad

d_capacidad_contenedor = range(25, 35, 1)
d_capacidad_semiremolque = range(400, 451, 5)

resultados = np.zeros((len(d_capacidad_contenedor), len(d_capacidad_semiremolque)))

for capacidad_contenedor in d_capacidad_contenedor:
    for capacidad_semiremolque in d_capacidad_semiremolque:
        beneficio_maximo, indices, cantidades = calcular_solucion_optima_con_costos(
            capacidad_contenedor, capacidad_semiremolque, productos
        )
        resultados[
            d_capacidad_contenedor.index(capacidad_contenedor),
            d_capacidad_semiremolque.index(capacidad_semiremolque),
        ] = beneficio_maximo

plt.figure(figsize=(8, 6))
plt.imshow(resultados, cmap="viridis", origin="lower")
plt.colorbar(label="Beneficio máximo")
plt.xlabel("Capacidad semirremolque (ton)")
plt.ylabel("Capacidad contenedor (m³)")
plt.xticks(range(len(d_capacidad_semiremolque)), d_capacidad_semiremolque)
plt.yticks(range(len(d_capacidad_contenedor)), d_capacidad_contenedor)
plt.tight_layout()
plt.savefig("sensibilidad.png", dpi=300)
plt.close()

import numpy as np
import matplotlib.pyplot as plt

# Parámetros para análisis
cambios_dimensiones = [0.8, 1.0, 1.2]  # Reducción o aumento del tamaño del producto
costo_fijo_transporte = [0, 50, 100]  # Costos fijos por producto

# Resultados para graficar
beneficios_por_cambio = np.zeros((len(cambios_dimensiones), len(costo_fijo_transporte)))

for i, cambio in enumerate(cambios_dimensiones):
    productos_modificados = [
        {
            "peso": p["peso"] * cambio,
            "volumen": p["volumen"] * cambio,
            "beneficio": p["beneficio"],
            "min_cant": p["min_cant"],
            "nombre": p["nombre"],
        }
        for p in productos
    ]
    for j, costo in enumerate(costo_fijo_transporte):
        beneficio, _, _ = calcular_solucion_optima_con_costos(
            capacidad_contenedor, capacidad_semiremolque, productos_modificados, costo
        )
        beneficios_por_cambio[i, j] = beneficio

# Generar gráficos
fig, ax = plt.subplots(figsize=(10, 6))

# Crear una curva para cada cambio en dimensiones
for i, cambio in enumerate(cambios_dimensiones):
    ax.plot(
        costo_fijo_transporte,
        beneficios_por_cambio[i, :],
        marker="o",
        label=f"Dimensiones x{cambio}",
    )

# Configurar el gráfico
ax.set_xlabel("Costo fijo de transporte ($)")
ax.set_ylabel("Beneficio máximo ($)")
ax.legend(title="Escala de dimensiones")
ax.grid(True, linestyle="--", alpha=0.7)

# Mostrar gráfico
plt.tight_layout()
plt.savefig("sensibilidad_costos.png", dpi=300)
plt.close()
