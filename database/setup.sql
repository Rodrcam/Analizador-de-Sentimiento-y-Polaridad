CREATE TABLE expresiones_raw (
    id_expresion SERIAL PRIMARY KEY,
    id_tuit VARCHAR(255) UNIQUE NOT NULL,
    texto_original TEXT NOT NULL,
    fecha_recoleccion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    etiqueta_tono VARCHAR(50) NOT NULL,
    etiqueta_intencion VARCHAR(50) NOT NULL
);

---

CREATE TABLE resultados_ia (
    id_clasificacion SERIAL PRIMARY KEY,
    id_tuit VARCHAR(255) UNIQUE NOT NULL,
    texto_procesado TEXT NOT NULL,
    prediccion_tono VARCHAR(50) NOT NULL,
    confianza_tono NUMERIC(5, 4) NOT NULL,
    prediccion_intencion VARCHAR(50) NOT NULL,
    fecha_clasificacion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_tuit) REFERENCES expresiones_raw (id_tuit)
);