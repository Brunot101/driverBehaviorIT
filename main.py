import pandas as pd
from information import ExtractInformation, choose_embedded_dimension, join_parts
import os

def main():
   
    url = "https://raw.githubusercontent.com/cephasax/OBDdatasets/refs/heads/master/masterDegreeResearch/19drivers.csv"
    df = pd.read_csv(url, sep=";")

   
    colluns = [
        'ENGINE_COOLANT_TEMP',
        'ENGINE_LOAD',
        'ENGINE_RPM',
        'INTAKE_MANIFOLD_PRESSURE',
        'Short Term Fuel Trim Bank 1',
        'MAF',
        'SPEED',
        'THROTTLE_POS',
        'TIMING_ADVANCE'
    ]
    df_19_final_features = df[colluns]

    df_19_final_features = df_19_final_features[~df_19_final_features.apply(lambda row: row.astype(str).str.contains('1:603').any(), axis=1)]


    #Removendo Unidades de medidas
    df_19_final_features['ENGINE_COOLANT_TEMP'] = df_19_final_features['ENGINE_COOLANT_TEMP'].str.replace('C', '', regex=False)
    df_19_final_features['ENGINE_LOAD'] = df_19_final_features['ENGINE_LOAD'].str.replace('%', '', regex=False)
    df_19_final_features['ENGINE_RPM'] = df_19_final_features['ENGINE_RPM'].str.replace('RPM', '', regex=False)
    df_19_final_features['INTAKE_MANIFOLD_PRESSURE'] = df_19_final_features['INTAKE_MANIFOLD_PRESSURE'].str.replace('kPa', '', regex=False)
    df_19_final_features['Short Term Fuel Trim Bank 1'] = df_19_final_features['Short Term Fuel Trim Bank 1'].str.replace('%', '', regex=False)
    df_19_final_features['MAF'] = df_19_final_features['MAF'].str.replace('g/s', '', regex=False)
    df_19_final_features['SPEED'] = df_19_final_features['SPEED'].str.replace('km/h', '', regex=False)
    df_19_final_features['THROTTLE_POS'] = df_19_final_features['THROTTLE_POS'].str.replace('%', '', regex=False)
    df_19_final_features['TIMING_ADVANCE'] = df_19_final_features['TIMING_ADVANCE'].str.replace('%', '', regex=False)
    df_19_final_features.head()

    #Convertendo para float após trocar a vírgula por ponto, e remover os dois pontos da parte de timestamp:
    df_19_final_features['ENGINE_COOLANT_TEMP'] = df_19_final_features['ENGINE_COOLANT_TEMP'].astype(float)
    df_19_final_features['ENGINE_LOAD'] = df_19_final_features['ENGINE_LOAD'].str.replace(',','.', regex = False).astype(float)
    df_19_final_features['ENGINE_RPM'] = df_19_final_features['ENGINE_RPM'].astype(float)
    df_19_final_features['INTAKE_MANIFOLD_PRESSURE'] = df_19_final_features['INTAKE_MANIFOLD_PRESSURE'].astype(float)
    df_19_final_features['Short Term Fuel Trim Bank 1'] = df_19_final_features['Short Term Fuel Trim Bank 1'].str.replace(',','.', regex = False).astype(float)
    df_19_final_features['MAF'] = df_19_final_features['MAF'].str.replace(',','.', regex = False).astype(float)
    df_19_final_features['SPEED'] = df_19_final_features['SPEED'].astype(float)
    df_19_final_features['THROTTLE_POS'] = df_19_final_features['THROTTLE_POS'].str.replace(',','.', regex = False).astype(float)
    df_19_final_features['TIMING_ADVANCE'] = df_19_final_features['TIMING_ADVANCE'].str.replace(',','.', regex = False).astype(float)

    for col in colluns:
        if df_19_final_features[col].dtype in ['float64', 'int64']:
            media = df_19_final_features[col].mean()
            df_19_final_features[col].fillna(media, inplace=True)


    df = df_19_final_features[colluns].dropna().reset_index(drop=True)

    
    window_length = 60 
    dx = choose_embedded_dimension(window_length)
    threads = 4

  
    path_out_prefix = f"saida/19drivers_window{window_length}_dx{dx}"
    os.makedirs("saida", exist_ok=True)

   
    extrator = ExtractInformation(
        df=df,
        path_out=path_out_prefix,
        window_length=window_length,
        embedding_dimension=dx,
        number_of_threads=threads
    )

    print("Extraindo informações com multiprocessing...")
    processes = extrator.run()
    for i, p in enumerate(processes, start=1):
        print(f"[{i}/{len(processes)}] Aguardando thread terminar...")
        p.join()

    print("Unindo arquivos finais...")
    join_parts(
    number_of_threads_per_file=threads,
    path_out= path_out_prefix,                
    path_out_time=path_out_prefix + ".csv.time"
    )
    os.rename(path_out_prefix, path_out_prefix + ".csv")
    print("Concluído")

if __name__ == "__main__":
    main()
