#importe de librerias
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel
#from pepe import recomendacion_juego
#from pepe import developer
app = FastAPI()



@app.get("/")
def read_root():
    a="Hola Mundo"
    return {a}



#cargo el dataset limpio
df = pd.read_json('datasets\datos.json')

#transformo las columnas a srt para poder tabajarlas
generos=df['genres'].astype(str)
tags=df['tags'].astype(str)
specs=df['specs'].astype(str)


vec=TfidfVectorizer()

#creo matriz
vec_matrix1= vec.fit_transform(generos)
vec_matrix2= vec.fit_transform(tags)
vec_matrix3= vec.fit_transform(specs)

#uno las matrises
matrix_completa=np.column_stack([vec_matrix1.toarray(),vec_matrix3.toarray(),vec_matrix2.toarray()])

#calculo la similituid del coseno
coseno=cosine_similarity(matrix_completa)




def recomendacion_juego( id_producto ):
    #id_producto=248820.0
    #buaca el juego en el dataFrame
    juego_en_data= df[df['id']== id_producto]

    if not juego_en_data.empty:
        juego_indice=juego_en_data.index[0]
        #obtengo los similares
        juegos_similares= coseno[juego_indice]
        #los ordeno de mayor a menor
        juegos_mas_similares=np.argsort(-juegos_similares)
        #obtengo los 6 primeros
        top_5_juegos=df.loc[juegos_mas_similares[0:6],'app_name']
        #los combierto en lista 
        top_5_juegos_mostrar=top_5_juegos.to_numpy().tolist()
        #tomo quito el primer valor para guardarlo en una variable para mostrar el nombre del juego que ingrese por id
        nombre_del_juego= top_5_juegos_mostrar.pop(0)
        a= (f'los 5 juegos recomendados para el id {id_producto} ({nombre_del_juego}) son: {top_5_juegos_mostrar}' )
        #return print(f'los 5 juegos recomendados para el id {id_producto} ({nombre_del_juego}) son: {top_5_juegos_mostrar}' )
        return a
    else:
        a='el juego no esta en la base de datos'
        return a

#creo un df con las variables a trabajar
df2= df[['release_date','price','developer','id']]
#paso todos precios a float, los que son F2P pasan a NaN
df2.loc[:,'price']=pd.to_numeric(df2['price'], errors='coerce')
#paso los NaN a 0
df2.loc[:,'price']=df2['price'].fillna(0)
#paso todo a minusculas
df2.loc[:,'developer'] = df2['developer'].str.lower()
#paso la columna a a formato de fecha
#df2.loc[:,'release_date'] = pd.to_datetime(df2['release_date'],errors='coerce')
df3=df2.copy()
#paso la columna a a formato de fecha
df3['release_date'] = pd.to_datetime(df2['release_date'],errors='coerce')

def developer2(desarrollador):
     # Filtro el DataFrame por el desarrollador especificado
    df_desarrollador = df2[df2['developer'] == desarrollador].copy()
    #me aseguro que la columna este en fomato fecha
    df_desarrollador['release_date'] = pd.to_datetime(df_desarrollador['release_date'], errors='coerce') 
    df_desarrollador.loc[:,'año'] = df_desarrollador['release_date'].dt.year
    
    # Agruparpo por año
    agrupado = df_desarrollador.groupby('año')
    
    # Calculo la cantidad de ítems por año
    Cantidad_Items = agrupado.size()
    
    # Calculo la cantidad de ítems free por año
    cantidad_free = agrupado.apply(lambda x: (x['price'] == 0.00).sum())
    
    # Calcular el porcentaje de ítems con precio cero por año
    cantidad_free_porsentaje = (cantidad_free / Cantidad_Items) * 100
    
    # Crear un DataFrame con los resultados
    resultado = pd.DataFrame({'Cantidad de Items': Cantidad_Items,'Contenido Free': cantidad_free_porsentaje}).reset_index()

    resultado['Contenido Free'] = resultado['Contenido Free'].apply(lambda x: f'{x:.2f}%')
    #results = results.reset_index(drop=True)
    resultado_final=resultado.to_dict(orient='index')
    return resultado_final




@app.get("/items/{item_id}")
def read_item(item_id: float):
    return {'respuesta':recomendacion_juego(item_id)}


@app.get("/developer/{desarrollador}")
def developer(desarrollador:str):
    desarrollador = desarrollador.lower()
    resultados = developer2(desarrollador)
    return{desarrollador:resultados}