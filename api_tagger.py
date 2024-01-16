import flask
from flask import request
import pandas as pd 
import numpy as np
import h2o

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Testing: http://localhost:5000/
@app.route('/', methods=['GET'])
def home():
    
    global data, df, criteria, clusters
    
    # Leer csv con datos y cargar en el dataframe data
    # Datos utilizados:
    ## nivel: mapeo directo
    ## entidad: mapeo directo
    ## moneda: mapeo directo
    ## modalidad: mapeo directo
    ## categoria: mapeo directo
    ## categoria_proveedor: calculada
    ## plurianual: mapeo directo
    ## adreferendum: mapeo directo
    ## subasta: mapeo directo
    ## forma_adjudicacion: mapeo directo
    ## monto_referencial: mapeo directo
    ## garantia: mapeo directo
    ## visita_al_sitio_no_nulo: calculada
    ## nivel_1: mapeo directo
    ## periodo_consulta: calculada
    ## periodo_llamado: calculada
    
    #leer CSV de datos, clusters entrenados y criterio de clasificacion
    data = pd.read_csv("data.csv")
    clusters = pd.read_csv("tagger_cluster_centers.csv")
    criteria = pd.read_csv("tagger_criteria.csv").iloc[0,0]
    
    #agregar columnas calculadas y convertir tipos
    data['visita_al_sitio'] = data['visita_al_sitio'].astype(str)
    data['categoria_proveedor'] = data['categoria_dncp_ultimo_mypime']
    data['visita_al_sitio_no_nulo'] = data['visita_al_sitio']
    
    #calcular categoria proveedor: MIC. DNCP. DNCP historico
    for index, row in data.iterrows():
        categoria = row['categoria_mypime_actual'].upper()
        if categoria.find("SIN CATEGORI") == -1:
            data.loc[index, 'categoria_proveedor'] = categoria
        else:
            categoria = row['categoria_sicp'].upper()
            if categoria.find("SIN CATEGORI") == -1:
                data.loc[index, 'categoria_proveedor'] = categoria
            else:
                data.loc[index, 'categoria_proveedor'] = row['categoria_dncp_ultimo_mypime'].upper()
        
        if row['visita_al_sitio'] == 'nan':
            data.loc[index, 'visita_al_sitio_no_nulo'] = "NO"
        else:
            data.loc[index, 'visita_al_sitio_no_nulo'] = "SI"
    
    #replace null datetimes por 2000-01-01 00:00:00
    data['fecha_tope_consulta'] = data['fecha_tope_consulta'].fillna('2000-01-01 00:00:00')
    data['fecha_entrega_oferta'] = data['fecha_entrega_oferta'].fillna('2000-01-01 00:00:00')
    data['fecha_publicacion_llamado'] = data['fecha_publicacion_llamado'].fillna('2000-01-01 00:00:00')
    
    #calcular periodo de consulta 
    data['fecha_tope_consulta'] = pd.to_datetime(data['fecha_tope_consulta'])
    data['fecha_publicacion_llamado'] = pd.to_datetime(data['fecha_publicacion_llamado'])
    data['periodo_consulta'] = ((data['fecha_tope_consulta']-data['fecha_publicacion_llamado'])/np.timedelta64(1,'D')).astype(int)
    
    #calcular periodo de llamado
    data['fecha_entrega_oferta'] = pd.to_datetime(data['fecha_entrega_oferta'])
    data['fecha_publicacion_llamado'] = pd.to_datetime(data['fecha_publicacion_llamado'])
    data['periodo_llamado'] = ((data['fecha_entrega_oferta']-data['fecha_publicacion_llamado'])/np.timedelta64(1,'D')).astype(int)
    
    df = data[["nro_pac", "nivel", "entidad", "moneda", "modalidad", "categoria", "categoria_proveedor", 
          "plurianual", "adreferendum", "subasta", "forma_adjudicacion", "monto_referencial", 
          "garantia", "visita_al_sitio_no_nulo", "nivel_1", "periodo_consulta", "periodo_llamado"]]
    
    
    h2o.init()
    
    #form html 
    html =  '''
                <h1>MiPymes Tagger</h1>
                <p>Tagger API</p>
                <p>Criteria: ''' + str(criteria) + '''</p>
                <form action="/api/v1/predict">
                ID PAC: <input type="text" name="idpac" />
                <input type="submit" method="get" value="Calcular"  />
            '''
    return html


@app.route('/api/v1/predict', methods=['GET'])
def api_predict():
    
    #leer identificador de pac
    idpac = 0
    if 'idpac' in request.args.keys():
        if request.args.get('idpac').isnumeric():
            idpac = int(request.args.get('idpac'))
    dato = df[df['nro_pac'] == idpac]
    
    
    #predict classification
    predClass = h2o.mojo_predict_pandas(dato, "tagger_classification.zip", 
                               genmodel_jar_path = 'jar/h2o-genmodel-3.38.0.1.jar', 
                               classpath = 'jar/*')

    #aplicar criterio propio de clasificacion 
    if predClass.loc[0,'MIPYME'] >= criteria:
        predClass.loc[0,'predict'] = "MIPYME"

    #predict clustering
    predCluster = h2o.mojo_predict_pandas(dato, "tagger_clustering.zip", 
                                   genmodel_jar_path = 'jar/h2o-genmodel-3.38.0.1.jar', 
                                   classpath = 'jar/*')
    cluster = clusters.iloc[predCluster['cluster']]
    
    #prepare aggregated result
    result = cluster
    result.insert(0,'predict',predClass.loc[0,'predict'])
    result.insert(1,'MIPYME',predClass.loc[0,'MIPYME'])
    result.insert(2,'GRANDE',predClass.loc[0,'GRANDE'])
    
    return result.to_json(orient = "records")

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>No se encuentra el recurso.</p>", 404

app.run()
