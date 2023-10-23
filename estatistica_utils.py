from scipy import stats
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def get_percentils_series(s, divisao=4, rotulo="valores_percentil"):
    """
    Dado uma série, a função retorna os valores de cada percentil de acordo com a divisão passada.
     O padrão é 4, portanto os quartis serão mostrados. A função retorna um data frame com os percentils
     como index, a coluna os respectivos valores e como título da coluna, o parâmetro rotulo.
    :param s: Series com os dados para serem divididos.
    :param divisao: quantidade de divisões a serem mostradas. E.g., 4 quartis, 10 decis, 100 percentis.
    :param rotulo: rótulo da coluna com os valores dos percentis.
    :return: data frame com os percentils como index, a coluna os respectivos valores e como título
    da coluna, o parâmetro rotulo.
    """

    return s.quantile([i / (divisao) for i in range(1, (divisao))]).to_frame(rotulo)

def ks_test(df,atributo_comparacao,p_max=0.1):
    significativo={}
    atributos = list(df.columns)
    atributos.remove(atributo_comparacao)
    for atributo in atributos:
        a = df[(df[atributo_comparacao] == 1)][atributo]
        b= df[(df[atributo_comparacao] == 0)][atributo]
        stat, p = stats.ks_2samp(a, b)
        #compara
        # print(stat,p)
        if p < p_max:
            significativo[atributo] = {"k":stat,
                                       "p-valor":p}
    return significativo

def correlacoes_lineares_significativas(df, atributo_corr, p_max=0.1):
    significativo = {}
    for atributos_mm in df.drop(atributo_corr,axis=1).columns:
        b = df[atributos_mm].dropna()
        a = df.loc[df[atributos_mm].notna(), atributo_corr]
        try:
            stat, p = pearsonr(a, b)
            if p < p_max:
                significativo[atributos_mm] = {"corr": stat,
                                               "p-valor":p}
        except ValueError:
            print('Atributos contém NaNs')
            raise

    return significativo

def mann_whitney(df1,df2,label1,label2,p_max=0.01):
    dif=0
    tab_geral= {"Atributo": [],
                      "Wilcoxon": [],
                      "P-valor("+str(p_max)+")": [],
                      "Mediana_"+label1: [],
                      "Mediana_"+label2: [],
                    "Delta Mediana":[],
                      # "Significância": []}
                      "Maior Mediana": []}
    for atrib in df1.columns:
        stat, p = stats.mannwhitneyu(df1[atrib], df2[atrib])
        if p < p_max:
            if np.median(df1[atrib]) > np.median(df2[atrib]):
                tab_geral["Atributo"].append(atrib)
                tab_geral["Wilcoxon"].append(round(stat, 2))
                tab_geral["P-valor("+str(p_max)+")"].append(round(p, 2))
                tab_geral["Mediana_"+label1].append(round(df1[atrib].median(), 2))
                tab_geral["Mediana_"+label2].append(round(df2[atrib].median(), 2))
                tab_geral['Delta Mediana'].append(round(df1[atrib].median() - df2[atrib].median(), 2))
                tab_geral["Maior Mediana"].append(label1)
            else:
                tab_geral["Atributo"].append(atrib)
                tab_geral["Wilcoxon"].append(round(stat, 2))
                tab_geral["P-valor(" + str(p_max) + ")"].append(round(p, 2))
                tab_geral["Mediana_" + label1].append(round(df1[atrib].median(), 2))
                tab_geral["Mediana_" + label2].append(round(df2[atrib].median(), 2))
                tab_geral['Delta Mediana'].append(round(df1[atrib].median() - df2[atrib].median(), 2))
                tab_geral["Maior Mediana"].append(label2)
            dif=dif+1
    print("Número de atributos significativos:", dif)
    return pd.DataFrame(tab_geral)

def get_conversao_testes(df_mapeamento):
    """
    Retorna a quantidade de pessoas que realizaram cada um dos testes do Mindmatch
    :param df_mapeamento: base de dados
    :return: data frame com as conversões por teste e a conversão total
    """
    total_teste_perfil = df_mapeamento[df_mapeamento['perfil-Capacidade analítica'].notna()].shape[0]
    total_teste_cultura = df_mapeamento[df_mapeamento['Cultura pontuação'].notna()].shape[0]
    total_teste_social = df_mapeamento[df_mapeamento['Social'].notna()].shape[0]
    total_teste_motivacional = df_mapeamento[df_mapeamento['Motivacional'].notna()].shape[0]

    try:
        total_teste_raciocinio = df_mapeamento[df_mapeamento['Adaptivo Transformado'].notna()].shape[0]
    except KeyError:
        print('Coluna Adaptivo Transformado renomeada para Raciocínio')
        total_teste_raciocinio = df_mapeamento[df_mapeamento['Raciocínio'].notna()].shape[0]

    total_teste_nexa = df_mapeamento[df_mapeamento['NEXA-overall'].notna()].shape[0]
    total_todos_testes = df_mapeamento[df_mapeamento['Potencial Bruto'].notna()].shape[0]



    try:
        total_teste_raciocinio_simplificado = df_mapeamento[df_mapeamento['Raciocínio (simplificado)-score'].notna()].shape[0]
        total_teste_interesses = df_mapeamento[df_mapeamento['Interesses profissionais-artistic'].notna()].shape[0]
        df_conversao_testes = pd.DataFrame([total_teste_perfil,
                                            total_teste_cultura,
                                            total_teste_social,
                                            total_teste_motivacional,
                                            total_teste_raciocinio,
                                            total_teste_nexa,
                                            total_teste_raciocinio_simplificado,
                                            total_teste_interesses,
                                            total_todos_testes],
                                           columns=['Qtde.'],
                                           index=['Perfil', 'Cultura', 'Social', 'Motivacional', 'Raciocínio', 'NEXA',
                                                  'Raciocínio Simplificado', 'Interesses', 'Todos PB'])

    except KeyError:
        print('Bateria operacional não inclusa')
        df_conversao_testes = pd.DataFrame([total_teste_perfil,
                                            total_teste_cultura,
                                            total_teste_social, total_teste_motivacional, total_teste_raciocinio,
                                            total_teste_nexa,
                                            total_todos_testes],
                                           columns=['Qtde.'],
                                           index=['Perfil', 'Cultura', 'Social', 'Motivacional', 'Raciocínio', 'NEXA', 'Todos PB'])

    df_conversao_testes['%'] = (df_conversao_testes['Qtde.'] / df_mapeamento.shape[0])

    return df_conversao_testes