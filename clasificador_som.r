rm(list=ls())
gc()

# Librerías
# ==============================================================================
if (!require("data.table")) install.packages("data.table"); library("data.table")
if (!require("ggplot2"))    install.packages("ggplot2")   ; library("ggplot2")
if (!require("dplyr"))      install.packages("dplyr")     ; library("dplyr")
if (!require("gridExtra"))  install.packages("gridExtra") ; library("gridExtra")

# Lectura de los datos de trabajo
# ==============================================================================
# Se fija el directorio de los datasets
setwd("data/")
# Datset "circulo.csv"
clouds <- setDF(fread("clouds.csv"))
names(clouds) <- c("x1", "x2", "y")
clouds[,1:2] <- scale(clouds[,1:2])

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

# Graficar en 2D Clasificaciones
categorias <- function(xValid, clasePredicciones, title){
  # Scatterplot
  gg <- ggplot(xValid, aes(x1, x2, color = get(clasePredicciones))) +
    geom_point(shape = 16, size = 3, show.legend = FALSE) +
    theme_minimal() +
    scale_color_gradient(low = "#0091ff", high = "#f0650e") +
    ggtitle(title)
  return(gg)
}

# ==============================================================================
SOM <- function(train, clase, n, m, prob, numEpocasGrueso, numEpocasMedio, 
                numEpocasFino, tasaAprendizajeGrueso, tasaAprendizajeMedio, 
                tasaAprendizajeFino, sleep, saltoGrafico, seed){
    #' Construye una red SOM sobre los datos de entrenamiento.
    #'
    #' train: datos de entrenamiento.
    #' clase: nombre de la columna de clases.
    #' n: cantidad de neuronas en el eje x.
    #' m: cantidad de neuronas en el eje y.
    #' prob: porcentaje (0 a 1) del dataset de entrenamiento. 
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
    
    # Se crea la variable idtempo 
    train <- as.data.table(mutate(train, idtempo = row_number()))
    # Se divide el dataset en un porcentaje ("prob") para datos de entrenamiento 
    # y otro para datos de validación
    set.seed(seed)
    dtrain <- as.data.table(train %>%
                              group_by(!!as.name(clase)) %>%
                              sample_frac(prob) %>%
                              ungroup)
    dvalid <- as.data.table(anti_join(train, dtrain, by = "idtempo"))
    # Se eliminan columnas innecesarias
    dtrain[, idtempo := NULL]
    dvalid[, idtempo := NULL]
    # Se separa la clase de los datos de entrenamiento
    xTrain <- as.matrix(dtrain[, !clase, with = FALSE])
    yTrain <- as.matrix(dtrain[, get(clase)])
    xValid <- as.matrix(dvalid[, !clase, with = FALSE])
    yValid <- dvalid[, get(clase)]
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
    redRandom <- matrix(runif(n * m * ncol(xTrain), 0.1, 0.25), 
                        ncol = ncol(xTrain))
    # Se suma la matriz de valores aleatorios a la red para dar algo de 
    # aleatoriedad a los pesos de la misma
    red <- red + redRandom
    # Se calcula la cantidad de épocas totales
    numEpoca <- numEpocasGrueso + numEpocasMedio +  numEpocasFino
    # Se inicia la actualización de los pesos del mapa
    for (i in 1:numEpoca){
      cat(paste0("Epoca N°: ", i, "...", "\n"))
      # Se selecciona una observación aleatoria de la red
      entrada <- xTrain[sample(nrow(xTrain), 1),]
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
        plot(graficarRed(xTrain, red, n, m, i))
        Sys.sleep(sleep)
      }
    }
    # Clasificación del dataset de entrenamiento
    # --------------------------------------------------------------------------
    # Se crean una matriz y un vector para guardar valores
    # Hay una columna por cada k centro y una fila por cada observación en xTrain
    # Es decir, se calcula para cada observación su distancia a cada neurona de 
    # la red
    distanciaAneuronas <- matrix(0, nrow(xTrain), nrow(red)) 
    neuronaMasCercana  <- c()
    # Se calcula la distancia entre las neuronas del mapa y las observaciones
    for (i in 1:nrow(xTrain)){
      for (j in 1:nrow(red)){
        # Se trabaja con valores absolutos en lugar de distancia euclídea
        distanciaAneuronas[i,j] <- sum(abs(xTrain[i,] - red[j,]))
      }
    }
    # Se determina que observación está más cercana a cada neurona del mapa y se 
    # asigna a ella
    for (i in 1:nrow(xTrain)){
      neuronaMasCercana[i] <- which.min(distanciaAneuronas[i,])
    }
    # Se juntan las clases de las observaciones con la neurona a las que se 
    # ecuentran más cercana
    claseRed <- data.frame(cbind(neuronaMasCercana, yTrain))
    names(claseRed) <- c("neurona", "clase")
    # Se suma cuantas observaciones de cada clase tiene cada neurona del mapa
    claseMayor <- claseRed %>% 
      group_by(neurona, clase) %>% 
      summarise(cnt = n()) %>% 
      as.data.table()
    # Se filtra para cada neurona la clase mayoritaria (suma de cantidad de clases 
    # de las observaciones)
    claseNeuronas <- claseMayor %>% group_by(neurona) %>% slice(which.max(cnt))
    # Se seleccionan columnas de interés
    claseNeuronas <- claseNeuronas[, 1:2]
    # Se agrega un índice para hacer match
    claseNeuronas$id <- 1:nrow(claseNeuronas) 
    # Predecir sobre el dataset de validación
    # --------------------------------------------------------------------------
    # Hay neuronas que no tienen observaciones de xTrain, entonces se las elimina 
    # porque no se sabe a que categoría pertenecen
    redConClases <- red[claseNeuronas$neurona,]
    # Se crea una matriz y un vector para guardar valores
    distAneuronaValid <- matrix(0, nrow(xValid), nrow(redConClases))
    nMasCercanaValid  <- c()
    # Se calcula la distancia entre las neuronas del mapa y las observaciones de 
    # validación
    for (i in 1:nrow(xValid)){
      for (j in 1:nrow(redConClases)){
        # Se trabaja con valores absolutos en lugar de distancia euclídea
        distAneuronaValid[i,j] <- sum(abs(xValid[i,] - redConClases[j,]))
      }
    }
    # Se determina que observación está más cercana a cada neurona del mapa y se 
    # asigna a ella
    for (i in 1:nrow(xValid)){
      nMasCercanaValid[i] <- which.min(distAneuronaValid[i,])
    }
    # Match nMasCercanasValid con el id de claseNeuronas
    # El valor de neurona más cercana en valid corresponde al id en claseNeurona, 
    # se busca entonces a que el nro de neurona corresponde en el conjunto "red"
    nMasCercanaValidMatch <- claseNeuronas[match(nMasCercanaValid, claseNeuronas$id ), 
                                           "neurona"]
    # Se agrega la neurona más cercana al dataset de validación
    xValid <- data.frame(cbind(xValid, claseReal = yValid, 
                               nc = nMasCercanaValidMatch$neurona))
    # Se matchea la clase de cada neurona (definida en claseNeuronas) con la 
    # neurona asignada a cada observación de xValid
    xValid[,5] <- claseNeuronas[match(xValid$nc, claseNeuronas$neurona), "clase"]
    names(xValid) <- c("x1", "x2", "claseReal", "neuronaCercana", "clasePredicha")
    # Se verifica el porcentaje de aciertos obtenidos con los nuevos datos 
    aciertos <- sum(xValid$claseReal - xValid$clasePredicha == 0) / nrow(xValid)
    return(list("aciertos" = aciertos, "xValid" = xValid))
}

# EVALUAR CLASIFICADOR
# ==============================================================================
# ==============================================================================

# Prueba del algoritmo SOM sobre el dataset circulos.csv
# ==============================================================================

# TEST 1: malla 20 x 20 y tasa de aprendizaje gruesa 0.3
som <- SOM(train                 = clouds,
           clase                 = "y",
           n                     = 20, 
           m                     = 20, 
           prob                  = 0.5,
           numEpocasGrueso       = 1000, 
           numEpocasMedio        = 1000, 
           numEpocasFino         = 1000,
           tasaAprendizajeGrueso = 0.3, 
           tasaAprendizajeMedio  = 0.1, 
           tasaAprendizajeFino   = 0.03,
           sleep                 = 1,
           saltoGrafico          = 500,
           seed                  = 123)
# Se grafican los puntos coloreados por la clase predicha
titulo <- paste0("Categorías Predichas - Tasa de aciertos: ", som$aciertos, 
                 " - Malla: 20 x 20 ")
g1 <- categorias(xValid = som$xValid, clasePredicciones = "clasePredicha", 
                 title = titulo)
g2 <- categorias(xValid = som$xValid, clasePredicciones = "claseReal", 
                 title = "Categorías Reales")
grid.arrange(g1, g2, nrow = 1)


