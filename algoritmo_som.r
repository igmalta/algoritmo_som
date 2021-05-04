rm(list=ls())
gc()

# Librerías
# ==============================================================================
if (!require("data.table")) install.packages("data.table"); library("data.table")
if (!require("ggplot2"))    install.packages("ggplot2")   ; library("ggplot2")

# Lectura de los datos de trabajo
# ==============================================================================
# Se fija el directorio de los datasets
setwd("data/")
# Se carga el datset "circulo.csv"
circulo <- fread("circulo.csv")
# Se renombran las variables
names(circulo) <- c("x1", "x2")
# Se escalea las variables
circulo <- scale(circulo)

# Se carga el dataset "te.csv"
te <- fread("te.csv")
# Se renombran las variables
names(te) <- c("x1", "x2")
# Se escalan las variables
te <- scale(te)

# Función para encontrar la neurona más cercana a una entrada 
# ==============================================================================
neuronaCercana <- function(entrada, red){
  #' Se ingresa una entrada y se busca la neurona más cercana a ella.
  #'
  #' entrada: entrada elegida.
  #' red: conjunto de neuronas que forman la red o mapa.
  
  # Un número alto cualquiera para iniciar
  distanciaMinima <- 1000000  
  # Se calcula la distancia entre una entrada elegida y cada punto de la red 
  # neuronal
  for (i in 1:nrow(red)){
    # Cada peso de la red
    w = red[i, ]
    # Se calcula la distancia entre la entrada y un punto de la red
    distancia <- sum(abs(w - entrada)) 
    if (distancia < distanciaMinima){
      distanciaMinima <- distancia
      indiceGanadora  <- i
    }
  }
  # Se guarda la neurona más cercana
  ganadora <- red[indiceGanadora,]
  return (list("ganadora" = ganadora, "indiceGanadora" = indiceGanadora))
}

# Funciones para graficar
# ==============================================================================
graficarRed <- function(train, red, n, m, epoca) {
  #' Grafica la red SOM en 2D.
  #' 
  #' train: dataset de entrenamiento.
  #' red: red de neuronas.
  #' n: cantidad de neuronas en el eje x o ancho.
  #' m: cantidad de neuronas en el eje y o alto.
  
  # Se convierte a data.table
  train <- data.table(train)
  red   <- data.table(red)
  # Se crea un id para trabajar
  red[, ID := .I]
  # Cantidad de grupos de polígonos 
  cntGrupos <- (n-1) * (m-1)
  # Se inicializa un data.frame vacío
  dfGrupos <- data.frame(matrix(NA, ncol = 4, nrow = m*(n-1)-1))
  # Se crea un vector de índices donde inician los polígonos
  vertices <- 1:(m*(n-1)-1)
  # Se crea un vector con los índices que sobran
  eliminar <- c()
  for (i in 1:(n-2)){
    eliminar <- c(eliminar, i*m) 
  }
  # Se eliminan  los vértices que no inician polígonos
  vertices <- vertices[-eliminar]
  # Cada polígono se compone de 4 vértices que siguen una secuencia para 
  # graficarlos
  for (i in vertices){
    dfGrupos[i,] <- c(i, i + 1, (i + m) + 1, i + m)
  }
  # Se eliminan filas sobrantes
  dfGrupos <- data.table(na.omit(dfGrupos))
  # Se crea un identificador por grupo de polígonos
  dfGrupos[, GRUPO := .I]
  # Se hace un reshape del data.frame
  reshape <- melt(dfGrupos, id.vars = c("GRUPO"))
  # Se eliminan variables que no son de interés
  reshape[, variable := NULL]
  # Se renombran variables
  names(reshape) <- c("GRUPO", "ID")
  # Se hace un join de los datasets
  red <- reshape[red, on = "ID"]
  names(red) <- c("grupo", "id", "x", "y")
  # Se ordenan los datos por grupo
  setorder(red, grupo)
  # Se cambia la posición de valores sumando y restando una unidad a id
  # Se hace para acomodar la secuencia necesaria para graficar los polígonos de 
  # la red
  for (i in 1:cntGrupos){
    red[((i*4)-1), id := id + 1]
    red[(i*4)    , id := id - 1]
  }
  # Se ordenan los datos por id
  setorder(red, id)
  # Se grafica
  gg <- ggplot() +
    geom_point(data = train, aes(x = x1, y = x2), color = "#2cb1c9") +
    geom_polygon(data = red, aes(x = x, y = y, group = grupo), color ="grey8", 
                 alpha = 0) +
    geom_point(data = red, aes(x = x, y = y), color = "red", size=3)  +
    xlab(paste0("Época N°: ", epoca))
    theme_minimal()
  return(gg)
}

# Gráfica para SOM unidimensional
graficarRed1D <- function(train, red, epoca) {
  #' Grafica un SOM unidimensional.
  #' 
  #' train: datos de entrenamiento.
  #' red: neuronas de la red unidimensional.
  
  # Se crea una secuencia para graficar la red (lineas) entre las neuronas
  # Se grafica
  gg <- ggplot() +
    geom_point(data = data.frame(train), aes(x = x1, y = x2, color = "red" )) +
    geom_point(data = data.frame(red),   aes(x = X1, y = X2, color = "blue", 
                                             size = 3)) +
    geom_line (data = data.frame(red),   aes(x = X1, y = X2)) +
    xlab(paste0("Época N°: ", epoca))
    theme_classic()
  return(gg)
}

# Algoritmo mapa autoorganizado (SOM)
# ==============================================================================
SOM <- function(train, n, m, numEpocasGrueso, numEpocasMedio, numEpocasFino,
                tasaAprendizajeGrueso, tasaAprendizajeMedio, tasaAprendizajeFino, 
                sleep, saltoGrafico, seed ){
  #' Construye una red SOM sobre los datos de entrenamiento.
  #'
  #' train: datos de entrenamiento.
  #' n: cantidad de neuronas en el eje x.
  #' m: cantidad de neuronas en el eje y.
  #' numEpocasGrueso: numéro de épocas en la etapa de ordenamiento topológico.
  #' numEpocasMedio: numéro de épocas en la etapa de transición.
  #' numEpocasFino: numéro de épocas en la etapa ajuste fino.
  #' tasaAprendizajeGrueso: tasa de aprendizaje en la etapa de ordenamiento 
  #' topológico.
  #' tasaAprendizajeMedio: tasa de aprendizaje en la etapa de transición.
  #' tasaAprendizajeFino: tasa de aprendizaje en la etapa de ajuste fino.
  #' sleep: tiempo en segundos que se pausa el algoritmo para graficar.
  #' saltoGrafico: cada cuantas épocas se grafica.
  #' seed: semilla.
  
  # Se crea una red de puntos (mapa) con forma regular, con distancia de 
  # separación unitaria
  red <- data.frame(cbind(rep(1:n, each = m), rep(seq(1:m), n)))
  # Se crea una matriz con los vecinos de cada neurona del mapa y se calcula la 
  # distancia de manhattan que los separa
  matrizVecinos <- as.matrix(dist(red, method = "manhattan"))
  # Se calcula para cada neurona de la red los vecinos que se encuentran dentro 
  # de una distancia manhattan 
  # aproximada a la mitad del mapa para utilizar luego en el ajuste grueso
  vecindadGruesa <- floor(( n + m)/4)
  indicesGrueso  <- apply(matrizVecinos, 1, 
                          function(x) (which(x <= vecindadGruesa)))
  # Se escalan los valores de la red para formar los pesos
  # Se suma 3 para descentrar la red
  red <- scale(red) + 3
  # Para una red unidimensional se alinean los puntos verticalmente sobre eje 
  # x = 0 (posición inicial)
  if (n == 1){
    red[,1] <- 0
  }
  # Se crea una matriz del mismo tamaño que el mapa pero con valores aleatorios 
  # en un determinado rango
  set.seed(seed)
  redRandom <- matrix(runif(n * m * ncol(train), 0.1, 0.25), ncol = ncol(train))
  # Se suma la matriz de valores aleatorios a la red para dar algo de 
  # aleatoriedad a los pesos de la misma
  red <- red + redRandom
  # Se calcula la cantidad de épocas totales
  numEpoca <- numEpocasGrueso + numEpocasMedio +  numEpocasFino
  # Se inicia la actualización de los pesos del mapa
  for (i in 1:numEpoca){
    cat(paste0("Epoca N°: ", i, "...", "\n"))
    # Se selecciona una observación aleatoria de la red
    entrada <- train[sample(nrow(train), 1),]
    # Se calcula la neurona ganadora, es decir, la más cercana a la entrada
    ganadora <- neuronaCercana(entrada, red)
    # Etapa de ordenamiento topológico
    if (i <= numEpocasGrueso){
      # Se actualizan pesos de la ganadora
      red[ganadora$indiceGanadora, ] <- red[ganadora$indiceGanadora,] + 
        tasaAprendizajeGrueso * (entrada - red[ganadora$indiceGanadora,])
      # Se actualizan pesos de los vecinos de la ganadora
      for (k in 1:length(indicesGrueso[[ganadora$indiceGanadora]])){
        indice <- indicesGrueso[[ganadora$indiceGanadora]][[k]]
        red[indice, ] <- red[indice, ] + tasaAprendizajeGrueso * 
          (entrada - red[indice, ])
      }
      # Etapa de transición  
    } else if (i < (numEpocasGrueso + numEpocasMedio)) {
      # La tasa de aprendizaje decrece linealmente
      tasaAprendizajeMedio <- tasaAprendizajeGrueso - 
        ((tasaAprendizajeGrueso - tasaAprendizajeMedio) / numEpocasMedio) * 
        (i - numEpocasGrueso)
      # El alcance de la vecindad decrece linelamente
      vecindadMedia <- 
        floor(vecindadGruesa - (vecindadGruesa - 1) / numEpocasMedio * 
                (i - numEpocasGrueso))
      indicesMedio  <- 
        apply(matrizVecinos, 1, function(x) (which(x <= vecindadMedia)))
      # Se actualizan pesos de la ganadora
      red[ganadora$indiceGanadora, ] <- red[ganadora$indiceGanadora, 1:2] + 
        tasaAprendizajeMedio * (entrada - red[ganadora$indiceGanadora, ])
      # Se actualizan pesos de los vecinos de la ganadora
      for (k in 1:length(indicesMedio[[ganadora$indiceGanadora]])){
        indice       <- indicesMedio[[ganadora$indiceGanadora]][[k]]
        red[indice, ] <- red[indice, ] + tasaAprendizajeMedio * 
          (entrada - red[indice, ])
      }
      # Etapa de ajuste fino
    } else {
      # Se actualizan pesos de la ganadora
      red[ganadora$indiceGanadora, ] <- red[ganadora$indiceGanadora, ] + 
        tasaAprendizajeFino * (entrada - red[ganadora$indiceGanadora, ])
    }
    # Graficar
    if (i %% saltoGrafico == 0 | i == 1 ){
      if (n == 1){
        plot(graficarRed1D(train, red, i))
        Sys.sleep(sleep)
      } else {
        plot(graficarRed(train, red, n, m, i))
        Sys.sleep(sleep)
      }
    }
  }
  return(list("mapa" = red))
}

# EVALUAR ALGORITMO
# ==============================================================================
# ==============================================================================

# Prueba sobre el dataset circulos.csv
# ------------------------------------------------------------------------------
# TEST 1: 20 x 20 y tasa de aprendizaje gruesa 0.9
som <- SOM(train                 = circulo, 
           n                     = 20, 
           m                     = 20, 
           numEpocasGrueso       = 200, 
           numEpocasMedio        = 500, 
           numEpocasFino         = 300,
           tasaAprendizajeGrueso = 0.9, 
           tasaAprendizajeMedio  = 0.1, 
           tasaAprendizajeFino   = 0.03,
           sleep                 = 1,
           saltoGrafico          = 200,
           seed                  = 123)

# TEST 2: 20 x 20 y tasa de aprendizaje gruesa 0.3
som <- SOM(train                 = circulo, 
           n                     = 20, 
           m                     = 20, 
           numEpocasGrueso       = 200, 
           numEpocasMedio        = 500, 
           numEpocasFino         = 300,
           tasaAprendizajeGrueso = 0.3, 
           tasaAprendizajeMedio  = 0.1, 
           tasaAprendizajeFino   = 0.03,
           sleep                 = 1,
           saltoGrafico          = 200,
           seed                  = 123)

# Prueba sobre el dataset te.csv
# ------------------------------------------------------------------------------
# TEST 1: 20 x 20 y tasa de aprendizaje gruesa 0.3
som <- SOM(train                 = te, 
           n                     = 20, 
           m                     = 20, 
           numEpocasGrueso       = 200, 
           numEpocasMedio        = 500, 
           numEpocasFino         = 300,
           tasaAprendizajeGrueso = 0.3, 
           tasaAprendizajeMedio  = 0.1, 
           tasaAprendizajeFino   = 0.03,
           sleep                 = 1,
           saltoGrafico          = 200,
           seed                  = 123)
