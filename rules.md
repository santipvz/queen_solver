# Descripción del juego
Queens es un juego de lógica visual donde debes colocar “reinas” (👑) en cada casilla de un tablero con regiones de colores, de modo que haya exactamente una reina en cada fila, columna y región coloreada, y ninguna se toquen ni tan siquiera en diagonal. Para resolverlo es clave combinar las restricciones de unicidad con la prohibición de adyacencia, aplicando técnicas de eliminación sistemática y reconocimiento de patrones en filas, columnas y regiones coloreadas.

## Objetivo

Colocar una reina (👑) en cada fila, cada columna y cada región coloreada del tablero, sin que dos reinas se toquen en ninguna de las ocho direcciones (horizontal, vertical o diagonal).
### 1.2. Componentes

- Tablero: una cuadrícula de tamaño variable (por ejemplo, 7×7, 8×8, etc.) dividida en regiones coloreadas de distinta forma y tamaño.

- Reinas: piezas representadas con el icono 👑. Algunas pueden venir pre-colocadas como pista inicial.

- Regiones coloreadas: agrupaciones de casillas delimitadas en el tablero, todas ellas con el mismo color de fondo.

### 2. Reglas básicas

- Unicidad por fila: exactamente una reina por cada fila.

- Unicidad por columna: exactamente una reina por cada columna.

- Unicidad por región: exactamente una reina dentro de cada región coloreada.

- No adyacencia: ninguna reina puede situarse en una casilla adyacente (incluidas diagonales) a otra reina.

### 3. Casos y patrones habituales

Para afrontar el puzzle, conviene reconocer patrones comunes y aplicar las reglas de forma sistemática:

#### 3.1. Regiones de tamaño 1

- Casilla única: si una región coloreada ocupa solo una casilla, ésta debe contener la reina obligatoriamente, pues es la única forma de cumplir la regla de unicidad de región.

#### 3.2. Regiones y filas/columnas casi completas

- Fila/columna casi completa: cuando en una fila o columna faltan pocas posiciones posibles (por ejemplo, todas las demás tienen ya vetos o reinas), la única casilla restante es la ubicación de la reina.

- Región de tamaño igual a número de filas libres: si una región abarca tantas casillas como el número de filas donde falta su reina, y estas casillas están en distintas filas, debes colocar una reinas en cada fila de esa región.

#### 3.3. Zonas de exclusión por adyacencia

- Una vez colocada una reina, marca como “vetadas” todas las casillas adyacentes en las ocho direcciones. Estas vetadas pueden reducir múltiples opciones en varias filas, columnas o regiones.

#### 3.4. Doble pista en “corchetes” (clue pairs)

- A veces el tablero incluye dos reinas pre-colocadas en la misma región o cruzadas por filas/columnas; estas pistas permiten propagar restricciones en líneas múltiples.

### 4. Pasos recomendados para resolver

A continuación, un flujo de trabajo paso a paso:

1. Análisis inicial

    - Identifica todas las regiones de tamaño 1 y coloca sus reinas.

    - Marca las casillas vetadas por adyacencia.

2. Recorrido de filas y columnas

    - Para cada fila y columna, cuenta cuántas reinas faltan y cuántas posiciones válidas existen; si coinciden, coloca reinas en esas posiciones.

    - Actualiza vetos tras cada colocación.

3. Recorrido de regiones

    - Para cada región coloreada, si solo queda una casilla sin reina ni veto, colócala.

    - En regiones más grandes, busca subregiones que hayan quedado con pocas casillas posibles.

4. Aplicación de deducciones cruzadas

    - Cruza la información de filas, columnas y regiones: una casilla puede quedar determinada porque es la única que puede albergar reina en fila y en región simultáneamente.

5. Chequeo de conflictos

    - Tras cada colocación, verifica que no se violen las reglas de unicidad ni de no-adjacencia.

6. Iteración

    - Repite los pasos 2–5 hasta completar el tablero.

##### Ejemplo de tablero
![tablero](board1.png)

##### Solución al tablero
![tablero_solved](board_solved.png)