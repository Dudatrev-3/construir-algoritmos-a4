import numpy as np
import pandas as pd

def calc_regressao(arquivo):

    df = pd.read_csv(arquivo)
    
    area = df["area_util_m2"].tolist()
    quartos = df["quantidade_quartos"].tolist()
    preco = df["preco_venda"].tolist()

    uns = [1]*len(area)
    X = np.column_stack((uns, area, quartos))
    Y = np.array(preco)

    a = np.transpose(X)
    b = np.dot(a, X)
    c = np.linalg.inv(b)
    d = np.dot(a, Y)
    beta = c @ d 
    return beta.tolist()
resultado = calc_regressao("imoveis_lar_perfeito.csv")
print(resultado)
