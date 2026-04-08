import pandas as pd
from multivariatedInformation import ExtractInformation, choose_embedded_dimension, join_parts
import os

def main():
   
    df = pd.read_csv("../artigo/cephas/df_19_final_features_sem_repeticao_sequencial.csv", sep=",")
    
    window_length = 60
    dx = 4
    threads = 4

    nomeDiretorio = f"../multivariated/cephas/cephasMultivariated{window_length}_{dx}"
    path_out_prefix = f"{nomeDiretorio}/cephasMultivariated{window_length}_dx{dx}"
    os.makedirs(nomeDiretorio, exist_ok=True)

   
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
